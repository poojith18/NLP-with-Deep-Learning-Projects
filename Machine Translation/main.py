import os
import sys
import torch
import transformers
import tensorflow as tf
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AdamWeightDecay
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from nltk.translate.bleu_score import sentence_bleu
np.random.seed(65)

def data_format_process(src_input_path,tgt_input_path,out_path,delimiter = '\t',encoding = 'utf-8'):
    with open(src_input_path, encoding=encoding) as f_source_in, \
         open(tgt_input_path, encoding=encoding) as f_target_in, \
         open(out_path, 'w', encoding=encoding) as f_out:
        for source_raw in f_source_in:
            source_raw = source_raw.strip()
            target_raw = f_target_in.readline().strip()
            if source_raw and target_raw:
                output_line = source_raw + delimiter + target_raw + '\n'
                f_out.write(output_line)

def data_preprocess(all_data):
    inps = all_data[src_lang]
    tgts = all_data[tgt_lang]
    inputs = tokenizer(inps, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgts, max_length=128, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

def sentence_blue_score(model, tokenizer, sample):
    src = sample[src_lang]
    tgt = sample[tgt_lang]
    input_ids = sample['input_ids']
    input_ids = torch.LongTensor(input_ids).view(1, -1)
    gen_ids = model.generate(input_ids)
    with tokenizer.as_target_tokenizer():
        pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return sentence_bleu([tgt.split()], pred.split(), weights=(1, 0, 0, 0))    

model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

src_lang = 'de'
tgt_lang = 'en'

data_files = {}
for split in ['train', 'val', 'test']:
    src_input_path = f'{split}.{src_lang}'
    tgt_input_path = f'{split}.{tgt_lang}'
    out_path = f'{split}.tsv'
    data_format_process(src_input_path, tgt_input_path, out_path)
    data_files[split] = [out_path]

dataset_dict = load_dataset('csv',delimiter='\t',column_names=[src_lang, tgt_lang],data_files=data_files)

data_tokenized = dataset_dict.map(data_preprocess, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

batch_size = 32
learning_rates = [0.0001, 0.00001, 0.000001]
weight_decay = 0.01

train_dataset = model.prepare_tf_dataset(data_tokenized["train"],batch_size=batch_size,shuffle=True,collate_fn=data_collator)

val_set_len = dataset_dict['val'].shape[0]
test_set_len = dataset_dict['test'].shape[0]
max_epochs = 10
best_bleu_score = 0.0
best_model = None
for lrate in learning_rates:
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    optimizer = AdamWeightDecay(learning_rate=lrate, weight_decay_rate=weight_decay)
    model.compile(optimizer=optimizer)
    for ep in range(1, max_epochs+1):
        print(f"Epoch {ep}")
        model.fit(train_dataset, epochs=1)
        val_bleu_score_list = [sentence_blue_score(model, tokenizer, data_tokenized['val'][i]) for i in range(val_set_len)]
        print(f'For learning rate {lrate}, The validation set BLEU Score: {np.mean(val_bleu_score_list)}')
        if np.mean(val_bleu_score_list) > best_bleu_score:
            best_bleu_score = np.mean(val_bleu_score_list)
            best_model = tf.keras.models.clone_model(model)

test_bleu_score_list = [sentence_blue_score(model, tokenizer, data_tokenized['test'][i]) for i in range(test_set_len)]
print(f'The test set BLEU Score: {np.mean(test_bleu_score_list)}')