import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab as vocab
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import state
from state import *
from evaluate import *
np.random.seed(65)

def read_file_data(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(' ||| ')
                if len(parts) == 3:
                    tokens = parts[0].split()
                    pos_tags = parts[1].split()
                    actions = parts[2].split()
                    sentences.append((tokens, pos_tags, actions))
    return sentences

def hidden_readfile_data(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(' ||| ')
                if len(parts) == 2:
                    tokens = parts[0].split()
                    pos_tags = parts[1].split()
                    sentences.append((tokens, pos_tags))
    return sentences

def read_tagset(file_path):
    tagset = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                tagset.append(line)
    return tagset

def read_pos_tags(file_path):
    pos_tags = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                pos_tags.append(line)
    return pos_tags

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        inp1 = sample[0]
        inp2 = sample[1]
        out = sample[2]
        return inp1, inp2, out

train_read_file_data = read_file_data("./../data/train.txt")
dev_read_file_data = read_file_data("./../data/dev.txt")
test_read_file_data = read_file_data("./../data/test.txt")
hidden_read_file_data = hidden_readfile_data("./../data/hidden.txt")
ques_read_file_data = hidden_readfile_data("./../data/ques.txt")
tagset_read_file_data = read_tagset("./../data/tagset.txt")
pos_set_read_file_data = read_pos_tags("./../data/pos_set.txt")
pos_set_word2ix = {word: index for index, word in enumerate(pos_set_read_file_data)}
tagset_word2ix = {word: index for index, word in enumerate(tagset_read_file_data)}
tagset_ix2word = {index: word for index, word in enumerate(tagset_read_file_data)}

#vec6b50d = vocab.GloVe(name='6B', dim=50)
#vec6b300d = vocab.GloVe(name='6B', dim=300)
#vec42b300d = vocab.GloVe(name='42B', dim=300)
vec840b300d = vocab.GloVe(name='840B', dim=300)

def preprocess(file_data):
    inputs_tokens = []
    inputs_pos_tokens = []
    outputs_tokens = []
    for i in range(len(file_data)):
        word_pos_dict = {file_data[i][0][n]: file_data[i][1][n] for n in range(len(file_data[i][0]))}
        word_pos_dict["[PAD]"] = 'NULL'
        buffer = file_data[i][0]+(["[PAD]","[PAD]"])
        t_parser = state.ParseState(["[PAD]","[PAD]"],buffer,[])
        tagset_ix = [tagset_word2ix[word] for word in file_data[i][2]]
        tokens = []
        while not is_final_state(t_parser,2):
            if len(t_parser.stack) >= 2 and len(t_parser.parse_buffer) >= 2:
                tokens = t_parser.stack[-2:] + t_parser.parse_buffer[0:2]
            elif len(t_parser.stack) < 2 and len(t_parser.parse_buffer) >= 2:
                padding_tokens = ["[PAD]"] * (2 - len(t_parser.stack))
                tokens = padding_tokens + t_parser.stack + t_parser.parse_buffer[0:2]
            elif len(t_parser.stack) >= 2 and len(t_parser.parse_buffer) < 2:
                padding_buffer = ["[PAD]"] * (2 - len(t_parser.parse_buffer))
                tokens = t_parser.stack[-2:] + t_parser.parse_buffer + padding_buffer
            else:
                padding_tokens = ["[PAD]"] * (2 - len(t_parser.stack))
                padding_buffer = ["[PAD]"] * (2 - len(t_parser.parse_buffer))
                tokens = padding_tokens + t_parser.stack + t_parser.parse_buffer + padding_buffer
            pos_tokens = [word_pos_dict[word] if word in word_pos_dict else "NULL" for word in tokens]
            pos_tokens_ix = [pos_set_word2ix[word] for word in pos_tokens]
            exclude_words = []
            good_tokens = [word for word in tokens if word not in exclude_words]
            token_embs = vec840b300d.get_vecs_by_tokens(good_tokens,lower_case_backup=True)
            # mean_token_embs = torch.mean(token_embs, dim=0)
            # inputs_tokens.append(mean_token_embs)
            token_embs = tuple(token_embs)
            cat_token_embs = torch.cat(token_embs, dim=0)
            inputs_tokens.append(cat_token_embs)
            inputs_pos_tokens.append(pos_tokens_ix)
            if len(file_data[i][2]) >0:
                act = (file_data[i][2]).pop(0)
            if act == "SHIFT":
                state.shift(t_parser)
            elif act.startswith("REDUCE_L"):
                state.left_arc(t_parser, act.split("_")[2])
            else:
                state.right_arc(t_parser, act.split("_")[2])    
            outputs_tokens.append(tagset_word2ix[act])
    return inputs_tokens,inputs_pos_tokens,outputs_tokens

(train_inputs_tokens,train_inputs_pos_tokens,train_outputs_tokens) = preprocess(train_read_file_data)
train_dataset_processed = []
for i in range(len(train_inputs_tokens)):
    l = []
    l.append(torch.tensor(train_inputs_tokens[i]))
    l.append(torch.tensor(train_inputs_pos_tokens[i]))
    l.append(torch.tensor(train_outputs_tokens[i]))
    train_dataset_processed.append(l)

train_dataset = CustomDataset(train_dataset_processed)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

def predict_actions(model, tokens, pos_tags):
    model.eval()
    with torch.no_grad():
        output = model(tokens, torch.tensor(pos_tags))   
    predicted_act_probs = torch.softmax(output, dim=1)
    predicted_actions = torch.argmax(predicted_act_probs, dim=1)
    return predicted_actions

def predict_actions_probs(model, tokens, pos_tags):
    model.eval()
    with torch.no_grad():
        output = model(tokens, torch.tensor(pos_tags))
    predicted_act_probs = torch.softmax(output, dim=1)    
    return predicted_act_probs

def dev_test_data_preprocess(file_data, model):
    inputs_tokens = []
    inputs_pos_tokens = []
    pred_action_list = [] 
    for i in range(len(file_data)):
        word_pos_dict = {file_data[i][0][n]: file_data[i][1][n] for n in range(len(file_data[i][0]))}
        word_pos_dict["[PAD]"] = 'NULL'
        buffer = file_data[i][0]+(["[PAD]","[PAD]"])
        t_parser = state.ParseState(["[PAD]","[PAD]"],buffer,[])
        tokens = []
        act_list = []
        while not is_final_state(t_parser,2):
            tokens = []
            if len(t_parser.stack) >= 2 and len(t_parser.parse_buffer) >= 2:
                tokens = t_parser.stack[-2:] + t_parser.parse_buffer[0:2]
            elif len(t_parser.stack) < 2 and len(t_parser.parse_buffer) >= 2:
                padding_tokens = ["[PAD]"] * (2 - len(t_parser.stack))
                tokens = padding_tokens + t_parser.stack + t_parser.parse_buffer[0:2]
            elif len(t_parser.stack) >= 2 and len(t_parser.parse_buffer) < 2:
                padding_buffer = ["[PAD]"] * (2 - len(t_parser.parse_buffer))
                tokens = t_parser.stack[-2:] + t_parser.parse_buffer + padding_buffer
            else:
                padding_tokens = ["[PAD]"] * (2 - len(t_parser.stack))
                padding_buffer = ["[PAD]"] * (2 - len(t_parser.parse_buffer))
                tokens = padding_tokens + t_parser.stack + t_parser.parse_buffer + padding_buffer
            pos_tokens = [word_pos_dict[word] if word in word_pos_dict else "NULL" for word in tokens]
            pos_tokens_ix = [pos_set_word2ix[word] for word in pos_tokens]
            token_embs = vec840b300d.get_vecs_by_tokens(tokens,lower_case_backup=True)
            # mean_token_embs = torch.mean(token_embs, dim=0)
            # pred_act_id = predict_actions(parser,mean_token_embs,[pos_tokens_ix])
            # pred_actions = predict_actions_probs(parser,mean_token_embs,[pos_tokens_ix])
            token_embs = tuple(token_embs)
            cat_token_embs = torch.cat(token_embs, dim=0)
            pred_act_id = predict_actions(model,cat_token_embs,[pos_tokens_ix])
            pred_actions = predict_actions_probs(model,cat_token_embs,[pos_tokens_ix])
            act = tagset_ix2word[np.array(pred_act_id)[0]]
            if len(t_parser.stack) <= 3:
                act = "SHIFT"
            elif len(t_parser.parse_buffer) <= 2:
                pred_actions = list(np.array(pred_actions).reshape(75))
                pred_actions.pop(0)
                prediction_act_id = pred_actions.index(max(pred_actions))
                act = tagset_ix2word[prediction_act_id+1]
            if act == "SHIFT":
                state.shift(t_parser)
            elif act.startswith("REDUCE_L"):
                state.left_arc(t_parser, act.split("_")[2])
            else:
                state.right_arc(t_parser, act.split("_")[2])
            act_list.append(act)
        pred_action_list.append(act_list)    
    return pred_action_list 

class TransitionParser(nn.Module):
    def __init__(self, demb, dpos, h, num_actions, pos_size):
        super(TransitionParser, self).__init__()
        self.word_embedding = nn.Linear(demb, h)
        self.pos_embedding = nn.Embedding(pos_size, dpos)
        self.final_pos_embedding = nn.Linear(dpos, h)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(h, num_actions)

    def forward(self, stack_buffer, pos_stack_buffer):
        word_embeds = self.word_embedding(stack_buffer)
        pos_embeds = self.pos_embedding(pos_stack_buffer)
        final_pos_embeds = self.final_pos_embedding(pos_embeds)
        pos_mean = torch.mean(final_pos_embeds, dim=1)
        hidden_rep = self.relu(torch.add(word_embeds, pos_mean))
        action_logits = self.linear(hidden_rep)
        return action_logits

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
max_epochs = 20
learning_rates = [0.01, 0.001, 0.0001]
loss_function = nn.CrossEntropyLoss()
best_perf_dict = {"metric": 0, "epoch": 0, "learning_rate":0}
for lrate in learning_rates:
    parser = TransitionParser(demb=1200, dpos = 50, h =200, num_actions = 75, pos_size = 18)
    optimizer = optim.Adam(parser.parameters(), lr=lrate)
    for ep in range(1, max_epochs+1):
        print(f"Epoch {ep}")
        train_loss = []
        for inp1, inp2, lab in tqdm(train_loader):
            parser.train()
            optimizer.zero_grad()
            out = parser(inp1, inp2)
            t_loss = loss_function(out, lab)
            t_loss.backward()
            optimizer.step()
            train_loss.append(t_loss.cpu().item())
        print(f"For learning rate {lrate}, the average training batch loss: {np.mean(train_loss)}")

        dev_data_action_list = dev_test_data_preprocess(dev_read_file_data, parser)
        dev_file_word_lists = [dev_read_file_data[i][0] for i in range(len(dev_read_file_data))]
        dev_file_gold_actions = [dev_read_file_data[i][2] for i in range(len(dev_read_file_data))]
        dev_uas_score, dev_las_score = compute_metrics(dev_file_word_lists, dev_file_gold_actions, dev_data_action_list,2)
        print(f"For learning rate {lrate}, the development LAS metric: {dev_las_score}\n")
        if dev_las_score > best_perf_dict["metric"]:
            best_perf_dict["metric"] = dev_las_score
            best_perf_dict["epoch"]  = ep
            best_perf_dict["learning_rate"]  = lrate
            torch.save({
                "model_param": parser.state_dict(),
                "optim_param": optimizer.state_dict(),
                "dev_metric": dev_las_score,
                "lr_metric": lrate,
                "epoch": ep
            }, f"/scratch/general/vast/u1405749/cs6957/assignment2/models/lr_{lrate}/model_{ep}")

print(f"""\nThe highest development LAS metric {best_perf_dict["metric"]} is with learning rate {best_perf_dict["learning_rate"]}.""")

lr_best = best_perf_dict["learning_rate"]
epoch_best = best_perf_dict["epoch"]
model_path = f"/scratch/general/vast/u1405749/cs6957/assignment2/models/lr_{lr_best}/model_{epoch_best}"
checkpoint = torch.load(model_path)

parser.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])

print(f"""Highest development LAS metric of loaded model: {checkpoint["dev_metric"]} with learning rate {checkpoint["lr_metric"]}""")

test_data_action_list = dev_test_data_preprocess(test_read_file_data, parser)
test_file_word_lists = [test_read_file_data[i][0] for i in range(len(test_read_file_data))]
test_file_gold_actions = [test_read_file_data[i][2] for i in range(len(test_read_file_data))]
test_uas_score, test_las_score = compute_metrics(test_file_word_lists, test_file_gold_actions, test_data_action_list, 2)
print("The best test UAS metric is :", test_uas_score)
print("The best test LAS metric is :", test_las_score)

hidden_data_action_list = dev_test_data_preprocess(hidden_read_file_data, parser)
with open('results.txt', 'w') as f:
    for li in hidden_data_action_list:
        line = " ".join(map(str, li))
        f.write(line + '\n')

ques_data_action_list = dev_test_data_preprocess(ques_read_file_data, parser)
with open('ans.txt', 'w') as f:
    for li in ques_data_action_list:
        line = " ".join(map(str, li))
        f.write(line + '\n')        