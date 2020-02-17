# BERT_NER
NER with BERT

Code to train NER model with BERT.

The following library is used. 

https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/

This uses pytorch.

Code could, for example, be run on the following dataset: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus

Training data should be in the format: 

Sentence #,Word,POS,Tag \
Sentence: 1,Thousands,NNS,O
,of,IN,O
,demonstrators,NNS,O
,have,VBP,O
,marched,VBN,O
,through,IN,O
,London,NNP,B-geo
,to,TO,O
,protest,VB,O
,the,DT,O
,war,NN,O
,in,IN,O
,Iraq,NNP,B-geo
,and,CC,O
,demand,VB,O
,the,DT,O
,withdrawal,NN,O
,of,IN,O
,British,JJ,B-gpe
,troops,NNS,O
,from,IN,O
,that,DT,O
,country,NN,O
,.,.,O
Sentence: 2,Families,NNS,O
,of,IN,O
,soldiers,NNS,O
,killed,VBN,O
,in,IN,O
,the,DT,O
...



*python train.py \
--data /notebook/nas-trainings/arne/OCCAM/NER_with_BERT/DATA/ner_dataset.csv \
--output_dir /notebook/nas-trainings/arne/OCCAM/NER_with_BERT/Fine_tuned_models/ner_en \
--epochs 5 \
--batch_size 64 \
--gpu 1 *
