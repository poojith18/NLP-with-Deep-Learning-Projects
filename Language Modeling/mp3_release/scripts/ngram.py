import numpy as np
from utils import *
import pickle
import math
np.random.seed(65)

with open ('./../data/vocab.pkl', 'rb') as f :
    vocab = pickle.load(f)

vocab_size = 386
train_file_list = get_files('./../data/train')
test_file_list = get_files('./../data/test')

train_file_data = convert_files(train_file_list, vocab)
test_file_data = convert_files(test_file_list, vocab) 

def create_ngrams(data, n):
    ngrams_dict = {}
    for li in data:
        list_without_padded = li.copy()
        list_padded = ['[PAD]'] * (n - 1) + list_without_padded
        for i in range(len(list_padded) - n + 1):
            ngrams_dict_key = list_padded[i : i + n - 1]
            ngrams_dict_key = tuple(ngrams_dict_key)
            if ngrams_dict_key not in ngrams_dict:
                s_dict = {}
                s_dict[list_padded[i + n - 1]] = 1
                ngrams_dict[ngrams_dict_key] = s_dict
            else:
                t_dict = ngrams_dict[ngrams_dict_key]
                if list_padded[i + n - 1] not in t_dict:
                    t_dict[list_padded[i + n - 1]] = 1
                else:
                    t_dict[list_padded[i + n - 1]] +=1   
    return ngrams_dict   

train_ngrams_dict = create_ngrams(train_file_data, 4)
print(f"The number of parameters of simple 4-gram model is: {len(train_ngrams_dict)}")
n_parameters = 0
for i in train_ngrams_dict.values():
    n_parameters += len(i)
#print(f"The number of parameters of simple 4-gram model is: {n_parameters}\n")

def test_perplexity(data, n, ngram_dict):
    perplexity_list = []
    for li in data:
        list_without_padded = li.copy()
        list_padded = ['[PAD]'] * (n - 1) + list_without_padded
        prob_list = []
        for i in range(len(list_padded) - n + 1):
            ngrams_dict_key = list_padded[i : i + n - 1]
            ngrams_dict_key = tuple(ngrams_dict_key)
            c_dict = ngram_dict.get(ngrams_dict_key,0)
            if c_dict!=0:
                n_val = c_dict.get(list_padded[i + n - 1],0)
                d_val = sum(c_dict.values())
            else:
                n_val = 0
                d_val = 0
            prob = (n_val + 1)/(d_val + vocab_size)
            prob_list.append(prob)
        log_prob_list = [math.log(p,2) for p in prob_list]
        neg_log_prob_list = [-1 * num for num in log_prob_list]
        avg_neg_prob_list = sum(neg_log_prob_list)/len(neg_log_prob_list)
        sen_perpelexity = 2**avg_neg_prob_list
        perplexity_list.append(sen_perpelexity)
    final_perplexity = sum(perplexity_list)/len(perplexity_list)
    return final_perplexity

test_perplexity_score = test_perplexity(test_file_data, 4, train_ngrams_dict)
print(f"The test set perplexity: {test_perplexity_score}")