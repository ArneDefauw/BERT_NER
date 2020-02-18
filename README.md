# BERT_NER

Named entity recognition with BERT

Code to finetune a pretrained BERT model for NER task.

The following library is used for loading pretrained BERT: 

https://github.com/huggingface/transformers

Code could, for example, be run on the following dataset: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus

Training data should be a csv, f.e.: 

Sentence #,Word,POS,Tag \
Sentence: 1,Thousands,NNS,O \
,of,IN,O \
,demonstrators,NNS,O \
,have,VBP,O \
,marched,VBN,O \
,through,IN,O  \
,London,NNP,B-geo  \
,to,TO,O  \
,protest,VB,O  \
,the,DT,O  \
,war,NN,O  \
,in,IN,O \
,Iraq,NNP,B-geo \
,and,CC,O \
,demand,VB,O \
,the,DT,O \
,withdrawal,NN,O \
,of,IN,O \
,British,JJ,B-gpe \
,troops,NNS,O \
,from,IN,O \
,that,DT,O \
,country,NN,O \
,.,.,O \
Sentence: 2,Families,NNS,O  \
,of,IN,O \
,soldiers,NNS,O \
,killed,VBN,O \
,in,IN,O \
,the,DT,O \
... 


One can finetune a pretrained BERT model on this data:

*python train.py \
--data /notebook/nas-trainings/arne/OCCAM/NER_with_BERT/DATA/ner_dataset.csv \
--output_dir /notebook/nas-trainings/arne/OCCAM/NER_with_BERT/Fine_tuned_models/ner_en \
--epochs 5 \
--batch_size 64 \
--gpu 1*

And next apply the finetuned model on a set of tokenized sentences:

*python test.py \
--input_file /notebook/nas-trainings/arne/OCCAM/NER_with_BERT/DATA/test_sentences.txt \
--model_dir /notebook/nas-trainings/arne/OCCAM/NER_with_BERT/Fine_tuned_models/ner_en \
--output_dir  /notebook/nas-trainings/arne/OCCAM/NER_with_BERT/output_folder \
--gpu 1*

This script will create a file *results* in the output folder with the NER results in the BIO scheme. 

F.e.:

sentence_0      thousands       O  \
sentence_0      of      O  \
sentence_0      demonstrators   O  \
sentence_0      have    O  \
sentence_0      marched O \
sentence_0      through O \
sentence_0      london  B-geo  \
sentence_0      to      O \
sentence_0      protest O \
sentence_0      the     O \
sentence_0      war     O \
sentence_0      in      O \
sentence_0      iraq    B-geo \
sentence_0      and     O \
sentence_0      demand  O \
sentence_0      the     O  \
sentence_0      withdrawal      O \
sentence_0      of      O \
sentence_0      british B-gpe  \
sentence_0      troops  O \
sentence_0      from    O \
sentence_0      that    O \
sentence_0      country O \
sentence_0      .       O \
sentence_1      iranian B-gpe \
sentence_1      officials       O \
sentence_1      say     O \
sentence_1      they    O \
sentence_1      expect  O \
sentence_1      to      O \
...




