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
                else:
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
c = 2
demb = 50
dpos = 50
h = 200
num_actions = 75
learning_rates = [0.01, 0.001, 0.0001]
pos_size = 18

train_read_file_data = read_file_data("./../data/train.txt")
dev_read_file_data = read_file_data("./../data/dev.txt")
test_read_file_data = read_file_data("./../data/test.txt")
hidden_read_file_data = read_file_data("./../data/hidden.txt")
tagset_read_file_data = read_tagset("./../data/tagset.txt")
pos_set_read_file_data = read_pos_tags("./../data/pos_set.txt")
pos_set_word2ix = {word: index for index, word in enumerate(pos_set_read_file_data)}
tagset_word2ix = {word: index for index, word in enumerate(tagset_read_file_data)}
tagset_ix2word = {index: word for index, word in enumerate(tagset_read_file_data)}

#vec6b50d = vocab.GloVe(name='6B', dim=50)
#vec6b300d = vocab.GloVe(name='6B', dim=300)
#vec42b300d = vocab.GloVe(name='42B', dim=300)
#vec840b300d = vocab.GloVe(name='840B', dim=300)

def glove_embeds(path):
    embeddings = {}
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            embedding_values = []
            for val in line.split(" ")[1:]:
                embedding_values.append(float(val))
            embeddings[line.split(" ")[0].strip()] = embedding_values
    return embeddings

globe6b50d_embeds = glove_embeds('./.vector_cache/glove.6B.50d.txt')
globe6b300d_embeds = glove_embeds('./.vector_cache/glove.6B.300d.txt')
globe42b300d_embeds = glove_embeds('./.vector_cache/glove.42B.300d.txt')
globe84b300d_embeds = glove_embeds('./.vector_cache/glove.840B.300d.txt')

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
            elif len(t_parser.parse_buffer) >= 2 and len(t_parser.parse_buffer) < 2:
                padding_buffer = ["[PAD]"] * (2 - len(t_parser.parse_buffer))
                tokens = t_parser.stack[-2:] + padding_buffer + t_parser.parse_buffer
            else:
                padding_tokens = ["[PAD]"] * (2 - len(t_parser.stack))
                padding_buffer = ["[PAD]"] * (2 - len(t_parser.parse_buffer))
                tokens = padding_tokens + t_parser.stack + padding_buffer + t_parser.parse_buffer
            pos_tokens = [word_pos_dict[word] if word in word_pos_dict else "NULL" for word in tokens]
            pos_tokens_ix = [pos_set_word2ix[word] for word in pos_tokens]
            exclude_words = ["[PAD]"]
            good_tokens = [word for word in tokens if word not in exclude_words]
            token_embs = [globe6b50d_embeds[word] for word in good_tokens]
            token_embs = np.array(token_embs)
            mean_token_embs = np.mean(token_embs, axis=0)
            inputs_tokens.append(mean_token_embs)
            inputs_pos_tokens.append(pos_tokens_ix)
            act = file_data[i][2].pop(0)
            if act == "SHIFT":
                state.shift(t_parser)
            elif act.startswith("REDUCE_L"):
                state.left_arc(t_parser, act.split("_")[2])
            else:
                state.right_arc(t_parser, act.split("_")[2])    
            outputs_tokens.append(tagset_word2ix[act])
    return inputs_tokens,inputs_pos_tokens,outputs_tokens

(train_inputs_tokens,train_inputs_pos_tokens,train_outputs_tokens) = preprocess(train_read_file_data)
#(dev_inputs_tokens,dev_inputs_pos_tokens,dev_outputs_tokens) = preprocess(dev_read_file_data)
#(test_inputs_tokens,test_inputs_pos_tokens,test_outputs_tokens) = preprocess(test_read_file_data)
train_dataset_processed = []

for i in range(len(train_inputs_tokens)):
    l = []
    l.append(torch.tensor(train_inputs_tokens[i]))
    l.append(torch.tensor(train_inputs_pos_tokens[i]))
    l.append(torch.tensor(train_outputs_tokens[i]))
    train_dataset_processed.append(l)

train_dataset = CustomDataset(train_dataset_processed)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# (hidden_inputs_tokens,hidden_inputs_pos_tokens) = hidden_data_preprocess(hidden_read_file_data)
# for i in range(len(hidden_inputs_tokens)):
#     l = []
#     l.append(torch.tensor(hidden_inputs_tokens[i]))
#     l.append(torch.tensor(hidden_inputs_pos_tokens[i]))
#     hidden_dataset_processed.append(l)
# hidden_dataset = CustomDataset(hidden_dataset_processed)
# hidden_loader = torch.utils.data.DataLoader(hidden_dataset, batch_size=64)

class TransitionParser(nn.Module):
    def __init__(self, demb, dpos, h, num_actions, pos_size):
        super(TransitionParser, self).__init__()
        self.word_embedding = nn.Linear(demb, h)
        self.pos_embedding = nn.Embedding(pos_size, dpos)
        self.final_pos_embedding = nn.Linear(dpos, h)
        self.linear = nn.Linear(h, num_actions)

    def forward(self, stack_buffer, pos_stack_buffer):
        word_embeds = self.word_embedding(stack_buffer)
        pos_embeds = self.pos_embedding(pos_stack_buffer)
        final_pos_embeds = self.final_pos_embedding(pos_embeds)
        pos_mean = torch.mean(final_pos_embeds, dim=1)
        hidden_rep = torch.add(word_embeds, pos_mean)
        action_probs = torch.softmax(self.linear(hidden_rep), dim=0)
        return action_probs

parser = TransitionParser(demb=50, dpos = 50, h =200, num_actions = 75, pos_size = 18)
optimizer = optim.Adam(parser.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
max_epochs = 3
for ep in range(1, max_epochs+1):
    print(f"Epoch {ep}")
    train_loss = []
    for inp1,inp2,lab in tqdm(train_loader):
        parser.train()
        optimizer.zero_grad()
        out = parser(inp1,inp2)
        t_loss = loss_function(out, lab)
        t_loss.backward()
        optimizer.step()
        train_loss.append(t_loss.cpu().item())
    print(f"For learning rate, the average training batch loss: {np.mean(train_loss)}")