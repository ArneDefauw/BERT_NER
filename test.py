import os
import pickle
import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForTokenClassification
from keras.preprocessing.sequence import pad_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Input-output:
    parser.add_argument("--input_file", dest="input_file",
                        help="path to the file with sentences on which to apply BERT-NER", required=True)
    parser.add_argument("--model_dir", dest="model_dir",
                        help="path to the directory where fine-tuned BERT model is stored", required=True)
    parser.add_argument("--output_dir", dest="output_dir",
                        help="path to the output folder", required=True)
    #Inference parameters:
    parser.add_argument("--batch_size", dest="batch_size", type=int ,default=32,
                        help="batch size used during training", required=False)
    parser.add_argument("--seq_length", dest="seq_length", type=int ,default=75,
                        help="maximum size of the (tokenized) sentences.", required=False)
    #GPU 0 or 1 used by pytorch:
    parser.add_argument("--gpu", dest="gpu", default=0, type=int,
                        help="gpu id", required=False)
    args = parser.parse_args()
    
    os.makedirs( args.output_dir , exist_ok=True) 
    
    #Load the sentences on which to apply BERT_NER:
    sentences=open( args.input_file  ).read().rstrip("\n").split( "\n"  )
    
    #Load the model and the data file with tags:
    with open( os.path.join( args.model_dir , "tags_vals" ) , "rb") as fp:
        tags_vals = pickle.load(fp)
    
    model = BertForTokenClassification.from_pretrained( args.model_dir , num_labels=len(tags_vals) )
    tokenizer = BertTokenizer.from_pretrained( args.model_dir )
    
    #Put the model on the GPU
    
    model.cuda(args.gpu) 

    model.eval()
    
    #tokenize the sentences:
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    #padding of sentences:
    test_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=args.seq_length, dtype="long", truncating="post" , padding="post")

    test_attention_masks = [[float(i>0) for i in ii] for ii in test_input_ids]

    #convert to pytorch tensors:
    test_input_ids = torch.tensor(test_input_ids)
    test_attention_masks = torch.tensor(test_attention_masks)

    #define dataloaders:
    test_data = TensorDataset( test_input_ids,  test_attention_masks )
    test_sampler = SequentialSampler(test_data)  #at test time, we pass the data sequentially
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size= args.batch_size )

    predictions=[]
    predictions_tags=[]

    for batch in test_dataloader:
        batch = tuple(t.cuda( args.gpu ) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        logits = logits[0].detach().cpu().numpy()

        predictions_batch = np.argmax(logits, axis=2)  #get the predicted labels

        for prediction_batch in predictions_batch:
            predictions.append(prediction_batch)
            pred_batch_tags=[tags_vals[ p ] for p in prediction_batch ]
            predictions_tags.append( pred_batch_tags )
    
    assert len(tokenized_texts) == len(predictions_tags)
    
    predictions_tags_no_pad=[]
    for sentence, prediction_tags in zip(tokenized_texts, predictions_tags  ):
        prediction_tags_no_pad=[]
        for i in range( len(sentence) ):
            prediction_tags_no_pad.append(prediction_tags[i])
        predictions_tags_no_pad.append(prediction_tags_no_pad)
    
    #write the tokenized sentence and the predicted tags to a tab separated file:
    with open(  os.path.join( args.output_dir , "results"  ) ,  "w"  ) as fp: 
        for i in range( len(tokenized_texts ) ):
            assert len( tokenized_texts[i]  ) == len( predictions_tags_no_pad[i] )
            for word, tag in zip( tokenized_texts[i], predictions_tags_no_pad[i]):
                fp.write( f"sentence_{i}\t{word}\t{tag}\n"  )  

