# Experimental code of multi-vector document-level dense retriever
This code repo contains the preprocess, training, inference and evaluation code for multi-vector document-level dense retriever on MSMARCO dataset. 

## Requirements
* Python 3.8.5
* PyTorch 1.9.1
* Transformers 4.2.5
* CUDA 11.1
* Faiss 1.7.2

## Perparation
First, download and unzip MSMARCO dataset from https://microsoft.github.io/msmarco/. Run script download_data.sh:
```
bash download_data.sh
```

## Preprocess
Prepare the padded tokens for document-level, paragraph-level, sentence-level dataset with preprocess.py:
```
python preprocess.py --data_type 0 --document_model roberta-base --paragraph_model cross-encoder/ms-marco-MiniLM-L-6-v2 --sentence_model cross-encoder/nli-MiniLM2-L6-H768 --output_dir #{preprocess_dir}
```

## Training
We use the [document STAR](https://drive.google.com/drive/folders/18GrqZxeiYFxeMfSs97UxkVHwIhZPVXTc?usp=sharing) as the warmup model. To produce the hard negatives for sentence-level, paragraph-level and the document-level training, run 
```
python ./star/prepare_hard.py --document_model ${warmup_model_path} --paragraph_model cross-encoder/ms-marco-MiniLM-L-6-v2 --sentence_model cross-encoder/nli-MiniLM2-L6-H768 --preprocess_dir ${preprocess_dir} --mode train --output_dir ${hardneg_path}
```
Then, we train the multi-vector document-level dense retriever with following hyperparameters:
```
python ./star/train.py --preprocess_dir ${preprocess_dir} --hardneg_path ${hardneg_path} --warmup_model #{warmup_model_path} --GAT_type RGAT
--do_train --optimizer_str adamw --learning_rate 2e-6 --gat_learning_rate 2e-3 --output_dir #{output_model_path}
```

##Inference
Obtain the relative score between documents and queries based on different document representations(document/paragraph/sentence):
```
python ./star/inference.py --mode dev  --model_path ${output_model_path} --document_type ${document_type} --topk 100
```

##Evaluation
Evaluate the trained model on MSMARCO dev document dataset:
```
python ./msmarco_eval.py ./data/doc/preprocess/dev-qrel.tsv ./data/doc/evaluate/adore-star/dev.rank.tsv 100
```
You will get:
| Model | Dev MRR@100 |
| ---------------- | ------------ |
| RoBERTa-STAR | 0.390 |
| RoBERTa-STAR + RGAT (doc) | 0.402 |
| RoBERTa-STAR + RGAT (passage)  | 0.406 |
| RoBERTa-STAR + RGAT (sentence)  | 0.396 |
