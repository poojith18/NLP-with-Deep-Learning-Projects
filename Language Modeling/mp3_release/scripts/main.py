import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import *
import os
import argparse
import pickle
from tqdm import tqdm
np.random.seed(65)

with open ('./../data/vocab.pkl', 'rb') as f :
    vocab = pickle.load(f)

train_file_list = get_files('./../data/train')
dev_file_list = get_files('./../data/dev')
test_file_list = get_files('./../data/test')

train_file_data = convert_files2idx(train_file_list, vocab)
dev_file_data = convert_files2idx(dev_file_list, vocab)
test_file_data = convert_files2idx(test_file_list, vocab)

train_file_token_data = convert_files(train_file_list, vocab)
token_dict = {}
for token_li in train_file_token_data:
    for i in range(len(token_li)):
        if token_li[i] not in token_dict:
            token_dict[token_li[i]] = 1
        else:
            token_dict[token_li[i]] += 1

weights_list = []
for token in vocab.keys():
    token_weight = 1 - (token_dict.get(token,0)/sum(token_dict.values()))
    weights_list.append(token_weight)

pad_index = vocab['[PAD]']
def inps_outs(data, subseq_length):
    inps = []
    outs = []
    for li in data:
        extra_pad = len(li)%subseq_length
        if extra_pad!=0:
            li = li + [pad_index] * (subseq_length + 1 - extra_pad)
        else:
            li = li + [pad_index]   
        for i in range(0, len(li) - subseq_length, subseq_length):
            inp_data = li[i : i + subseq_length]
            out_data = li[i + 1 : i + subseq_length + 1]
            inps.append(inp_data)
            outs.append(out_data)
    return inps, outs           

k = 500
train_inps, train_outs = inps_outs(train_file_data, k)

dev_inps, dev_outs = inps_outs(dev_file_data, k)
test_inps, test_outs = inps_outs(test_file_data, k)

train_inps = torch.tensor(train_inps, dtype=torch.long)
train_outs = torch.tensor(train_outs, dtype=torch.long)
dev_inps = torch.tensor(dev_inps, dtype=torch.long)
dev_outs = torch.tensor(dev_outs, dtype=torch.long)
test_inps = torch.tensor(test_inps, dtype=torch.long)
test_outs = torch.tensor(test_outs, dtype=torch.long)

train_dataset = TensorDataset(train_inps, train_outs)
dev_dataset = TensorDataset(dev_inps, dev_outs)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

vocab_size = 386
embedding_dim = 50
hidden_dim = 200
num_layers = 1

class CharacterLevelLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(CharacterLevelLanguageModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden, context):
        embedded = self.embedding(inputs)
        output, (hidden_state, context_vector) = self.lstm(embedded, (hidden, context))
        f_output1 = self.fc1(output)
        f_output1_relu = self.relu(f_output1)
        f_output2 = self.fc2(f_output1_relu)
        return f_output2, (hidden_state, context_vector) 

learning_rates = [0.0001, 0.00001, 0.000001]
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
max_epochs = 5
loss_function = nn.CrossEntropyLoss(weight = torch.tensor(weights_list).to(device), ignore_index = vocab['[PAD]'])
#loss_function = nn.CrossEntropyLoss(weight = torch.tensor(weights_list).to(device))
loss_function_eval = nn.CrossEntropyLoss(ignore_index = vocab['[PAD]'], reduction = 'none')
#loss_function_eval = nn.CrossEntropyLoss(reduction = 'none')
best_perf_dict = {"metric": 100, "epoch": 0, "learning_rate":0}
for lrate in learning_rates:
    model = CharacterLevelLanguageModel(vocab_size = vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim, num_layers = num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lrate)
    for ep in range(1, max_epochs+1):
        print(f"Epoch {ep}")
        train_loss = []
        hidden = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
        context = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
        for inp, lab in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            out, (hidden_state, context_vector) = model(inp.to(device), hidden, context)
            t_loss = loss_function(out.view(-1, vocab_size), lab.to(device).view(-1))
            t_loss.backward()
            optimizer.step()
            train_loss.append(t_loss.cpu().item())
        print(f"For learning rate {lrate}, the average training batch loss: {np.mean(train_loss)}")

        dev_perplexity_list = []
        for i in range(len(dev_inps)):
            dev_inp = dev_inps[i]
            dev_out = dev_outs[i]
            upd_dev_inp = torch.where(dev_inp != vocab['[PAD]'], 1.0, 0.0)
            len_wo_pad = torch.sum(upd_dev_inp)
            hidden_state = torch.zeros(num_layers, hidden_dim).to(device)
            context_vector = torch.zeros(num_layers, hidden_dim).to(device)
            model.eval()
            with torch.no_grad():
                d_out, (hidden_state, context_vector) = model(dev_inp.to(device), hidden_state, context_vector)
                d_loss = loss_function_eval(d_out.view(-1, vocab_size), dev_out.to(device).view(-1))
                d_avg_loss = torch.dot(upd_dev_inp.to(device),d_loss.to(device))/len_wo_pad.to(device)
                #d_perplexity = 2**(d_avg_loss.cpu().item())
                d_perplexity = 2.718**(d_avg_loss.cpu().item())
                dev_perplexity_list.append(d_perplexity)
        print(f"For learning rate {lrate}, the development set perplexity: {np.mean(dev_perplexity_list)}\n")

        if np.mean(dev_perplexity_list) < best_perf_dict["metric"]:
            best_perf_dict["metric"] = np.mean(dev_perplexity_list)
            best_perf_dict["epoch"]  = ep
            best_perf_dict["learning_rate"]  = lrate
            torch.save({
                "model_param": model.state_dict(),
                "optim_param": optimizer.state_dict(),
                "dev_metric": np.mean(dev_perplexity_list),
                "lr_metric": lrate,
                "epoch": ep
            }, f"/scratch/general/vast/u1405749/cs6957/assignment3/models/lr_{lrate}/model_{ep}")

lr_best = best_perf_dict["learning_rate"]
epoch_best = best_perf_dict["epoch"]
model_path = f"/scratch/general/vast/u1405749/cs6957/assignment3/models/lr_{lr_best}/model_{epoch_best}"
checkpoint = torch.load(model_path)

model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])

print(f"""Development set perplexity of loaded model: {checkpoint["dev_metric"]} with learning rate {checkpoint["lr_metric"]}""")

test_perplexity_list = []
for i in range(len(test_inps)):
    test_inp = test_inps[i]
    test_out = test_outs[i]
    upd_test_inp = torch.where(test_inp != vocab['[PAD]'], 1.0, 0.0)
    len_w_o_pad = torch.sum(upd_test_inp)
    hidden_state = torch.zeros(num_layers, hidden_dim).to(device)
    context_vector = torch.zeros(num_layers, hidden_dim).to(device)
    model.eval()
    with torch.no_grad():
        te_out, (hidden_state, context_vector) = model(test_inp.to(device), hidden_state, context_vector)
        te_loss = loss_function_eval(te_out.view(-1, vocab_size), test_out.to(device).view(-1))
        te_avg_loss = torch.dot(upd_test_inp.to(device),te_loss.to(device))/len_w_o_pad.to(device)
        #te_perplexity = 2**(te_avg_loss.cpu().item())
        te_perplexity = 2.718**(te_avg_loss.cpu().item())
        test_perplexity_list.append(te_perplexity)
print(f"The test set perplexity: {np.mean(test_perplexity_list)}\n")

num_param = sum(p.numel() for p in model.parameters())
print(f"The number of parameters of LSTM network with number of layers {num_layers} is: {num_param}\n")
    
idx_to_char = {v : k for k, v in vocab.items()}
def generate_text(seed_sequence, model, length):
    hidden_state = torch.zeros(num_layers, hidden_dim).to(device)
    context_vector = torch.zeros(num_layers, hidden_dim).to(device)
    model.eval()
    with torch.no_grad():  
        output_text = seed_sequence
        input_sequence = seed_sequence
        input_data = torch.tensor([vocab[c] for c in input_sequence], dtype=torch.long)
        output, (hidden_state, context_vector) = model(input_data.to(device), hidden_state, context_vector)
        input_char = torch.tensor(input_data[-1]).to(device)
        for i in range(length):
            output, (hidden_state, context_vector) = model(input_char.view(1), hidden_state, context_vector)
            probabilities = torch.softmax(output, dim=1)
            predicted_char = idx_to_char[torch.multinomial(probabilities, 1).item()]
            output_text += predicted_char
            input_sequence += predicted_char
            input_char = torch.tensor(vocab[predicted_char]).to(device)
    return output_text

seed_sequences = ["The little boy was", "Once upon a time in", "With the target in", "Capitals are big cities. For example,", "A cheap alternative to"]
outputs_of_seed_sequences = []
for i in seed_sequences:
    seed_sequence_output = generate_text(i, model, 200)
    outputs_of_seed_sequences.append(seed_sequence_output)

with open('seed_sequences_ans.txt', 'w') as f:
    for st in outputs_of_seed_sequences:
        f.write(st + '\n')
