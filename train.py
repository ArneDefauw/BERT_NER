import os
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from seqeval.metrics import f1_score
from tqdm import tqdm, trange
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME


class SentenceGetter():
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
        
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Input-output:
    parser.add_argument("--data", dest="data",
                        help="path to the training data", required=True)
    parser.add_argument("--output_dir", dest="output_dir",
                        help="path to the output folder", required=True)
    #Training parameters:
    parser.add_argument("--batch_size", dest="batch_size", type=int ,default=32,
                        help="batch size used during training", required=False)
    parser.add_argument("--seq_length", dest="seq_length", type=int ,default=75,
                        help="maximum size of the (tokenized) sentences", required=False)
    parser.add_argument("--epochs", dest="epochs", type=int ,default=5,
                        help="nr of training epochs", required=False)
    parser.add_argument("--val_size", dest="val_size", type=float ,default=0.1,
                        help="size of the validation set extracted from the training data", required=False)
    parser.add_argument("--seed", dest="seed", type=int ,default=2018,
                        help="seed for train test split", required=False)
    #GPU 0 or 1 used by pytorch:
    parser.add_argument("--gpu", dest="gpu", default=0, type=int,
                        help="gpu id", required=False)
    args = parser.parse_args()

    
    os.makedirs( args.output_dir , exist_ok=True) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    device_name=torch.cuda.get_device_name(0) 
    
    print(   'device: {}; numbers of gpus available {}. Name of the device: {}'.format(device , n_gpu , device_name ))
    
    data=pd.read_csv( args.data , encoding='latin'  )
    data=data.fillna( method='ffill'  )
    
    getter=SentenceGetter(data)
    
    sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
    labels = [[s[2] for s in sent] for sent in getter.sentences]

    tags_vals = list(set(data["Tag"].values))
    tags_vals=sorted( tags_vals )
    
    tag2idx = {t: i for i, t in enumerate(tags_vals)}
    
    #tokenize the sentences with BERT
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    
    #Convert the tokens in the tokenized sentences to ID's, and do some padding
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=args.seq_length, dtype="long", truncating="post" , padding="post")
    
    
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=args.seq_length, value=tag2idx["O"], padding="post",  #value 
                     dtype="long", truncating="post")   
    
    #create masks, so the padded elements are ignored in the sequence
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
    
    #train test split
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                                random_state=args.seed, test_size=args.val_size )
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=args.seed, test_size=args.val_size )
    
    #convert to pytorch tensors:
    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)
    
    #Define dataloaders:
    bs=args.batch_size
    
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)  #shuffle the data at training time
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)  #at test time, we pass the data sequentially
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
    
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))
    # BertForTokenClassification is a fine tuning model that wraps BertModel and adds a token-level classifier on top of the BertModel. Token level classifier is a linear layer that takes as input the last hidden 
    # state of the sequence. 
    
    #put the model on the GPU:
    
    model.cuda( args.gpu )
    
    #You can choose to only train the linear classifier on top of BERT, and keep all other weights fixed (FULL_FINETUNING=FALSE).
    #Add weight decay as regularization to the main weight matrices.
    FULL_FINETUNING = True  
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)   #==> Adam function adds the keys 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad' to the 'optimizer_grouped_parameters'
    
    
    #Train the model:
    
    epochs = args.epochs
    max_grad_norm = 1.0

    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.cuda( args.gpu ) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))  #This is the average loss per epoch.
        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.cuda( args.gpu ) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask, labels=b_labels)
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
        valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

    #Save the model, configuration and vocabulary that you have fine-tuned
    
    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    #model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join( args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir , CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)
    
    with open( os.path.join( args.output_dir , "tags_vals" ) , "wb" ) as fp:
        pickle.dump( tags_vals, fp  )
    
    