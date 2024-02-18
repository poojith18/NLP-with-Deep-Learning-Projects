import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import *
import os
import argparse
from tqdm import tqdm
np.random.seed(65)

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
args = parser.parse_args()
output_dir = args.output_dir

context_size = 5
word_to_idx = get_word2ix("./../vocab.txt")
train_list = get_files("./../data/train")
dev_list = get_files("./../data/dev")
train_process_data = process_data(train_list, context_size, word_to_idx)
dev_process_data = process_data(dev_list, context_size, word_to_idx)

def contexts_targets(data, context_size):
    contexts = []
    targets = []
    for li in data:
        for i in range(context_size, len(li) - context_size):
            context = li[i - context_size:i] + li[i + 1:i + context_size + 1]
            target = li[i]
            contexts.append(context)
            targets.append(target)
    return contexts, targets

train_contexts, train_targets = contexts_targets(train_process_data, context_size)
dev_contexts, dev_targets = contexts_targets(dev_process_data, context_size)

train_contexts = torch.tensor(train_contexts, dtype=torch.long)
train_targets = torch.tensor(train_targets, dtype=torch.long)
dev_contexts = torch.tensor(dev_contexts, dtype=torch.long)
dev_targets = torch.tensor(dev_targets, dtype=torch.long)

train_dataset = TensorDataset(train_contexts, train_targets)
dev_dataset = TensorDataset(dev_contexts, dev_targets)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

vocab = list(get_word2ix("./../vocab.txt").keys())
vocab_size = len(vocab)
embedding_dim = 100
learning_rates = [0.01,0.001,0.0001]

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embedded = self.embeddings(inputs)
        sum_embedded = torch.sum(embedded, dim=1)
        output = self.linear(sum_embedded)
        return output

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
max_epochs = 10
loss_function = nn.CrossEntropyLoss()
best_perf_dict = {"metric": 20, "epoch": 0, "learning_rate":0}
for lrate in learning_rates:
    model = CBOW(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lrate)
    for ep in range(1, max_epochs+1):
        print(f"Epoch {ep}")
        train_loss = []
        for inp, lab in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            out = model(inp.to(device))
            t_loss = loss_function(out, lab.to(device))
            t_loss.backward()
            optimizer.step()
            train_loss.append(t_loss.cpu().item())
        print(f"For learning rate {lrate}, the average training batch loss: {np.mean(train_loss)}")

        dev_loss = []
        for inp, lab in tqdm(dev_loader):
            model.eval()
            with torch.no_grad():
                out = model(inp.to(device))
                d_loss = loss_function(out, lab.to(device))
                dev_loss.append(d_loss.cpu().item())
        print(f"For learning rate {lrate}, the average development batch loss: {np.mean(dev_loss)}\n")
        if np.mean(dev_loss) < best_perf_dict["metric"]:
            best_perf_dict["metric"] = np.mean(dev_loss)
            best_perf_dict["epoch"]  = ep
            best_perf_dict["learning_rate"]  = lrate
            print(output_dir)
            torch.save({
                "model_param": model.state_dict(),
                "optim_param": optimizer.state_dict(),
                "dev_metric": np.mean(dev_loss),
                "lr_metric": lrate,
                "epoch": ep
            }, f"/scratch/general/vast/u1405749/cs6957/assignment1/models/lr_{lrate}/model_{ep}")

print(f"""\nThe lowest development set loss of {best_perf_dict["metric"]} is with learning rate {best_perf_dict["learning_rate"]}.""")

lr_best = best_perf_dict["learning_rate"]
epoch_best = best_perf_dict["epoch"]
model_path = f"/scratch/general/vast/u1405749/cs6957/assignment1/models/lr_{lr_best}/model_{epoch_best}"
checkpoint = torch.load(model_path)

model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])

print(f"""Development loss of loaded model: {checkpoint["dev_metric"]} with learning rate {checkpoint["lr_metric"]}""")

embeddings = list(model.linear.parameters())[0].detach().cpu().numpy()
with open('embeddings.txt', 'w') as embeddings_file:
    embeddings_file.write(f'{vocab_size} {embedding_dim}\n')
    for word, idx in word_to_idx.items():
        embedding = embeddings[idx]
        embeddings_file.write(f'{word} {" ".join(map(str, embedding))}\n')
