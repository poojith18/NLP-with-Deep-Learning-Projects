import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn import metrics
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
np.random.seed(42)

sst2_dataset = load_dataset("gpt3mix/sst2")
num_classes = 2

# model_name = "prajjwal1/bert-tiny"
# d_rep = 128
model_name = "prajjwal1/bert-mini"
d_rep = 256

tokenizer = BertTokenizer.from_pretrained(model_name)

train_data = tokenizer(sst2_dataset["train"]['text'], truncation=True, max_length = 512, padding='max_length', return_tensors='pt')
val_data = tokenizer(sst2_dataset["validation"]['text'], truncation=True, max_length = 512, padding='max_length', return_tensors='pt')
test_data = tokenizer(sst2_dataset["test"]['text'], truncation=True, max_length = 512, padding='max_length', return_tensors='pt')

input_ids_train = train_data['input_ids']
token_type_ids_train = train_data['token_type_ids']
attention_masks_train = train_data['attention_mask']
labels_train = torch.tensor(sst2_dataset["train"]['label'])

input_ids_val = val_data['input_ids']
token_type_ids_val = val_data['token_type_ids']
attention_masks_val = val_data['attention_mask']
labels_val = torch.tensor(sst2_dataset["validation"]['label'])

input_ids_test = test_data['input_ids']
token_type_ids_test = test_data['token_type_ids']
attention_masks_test = test_data['attention_mask']
labels_test = torch.tensor(sst2_dataset["test"]['label'])

train_dataset = TensorDataset(input_ids_train, token_type_ids_train, attention_masks_train, labels_train)
val_dataset = TensorDataset(input_ids_val, token_type_ids_val, attention_masks_val, labels_val)
test_dataset = TensorDataset(input_ids_test, token_type_ids_test, attention_masks_test, labels_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class BERTModel(nn.Module):
    def __init__(self, model_name, d_rep, num_classes):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc1 = nn.Linear(d_rep, num_classes)
    
    def forward(self, inputs):
        embeddings = self.bert(**inputs)
        cls_embeddings = embeddings.pooler_output
        output = self.fc1(cls_embeddings)
        return output

learning_rates = [0.0001, 0.00001, 0.000001]

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
max_epochs = 10
loss_function = nn.CrossEntropyLoss()
best_perf_dict = {"metric": 0, "epoch": 0, "learning_rate":0}
for lrate in learning_rates:
    model = BERTModel(model_name, d_rep, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lrate)
    for ep in range(1, max_epochs+1):
        print(f"Epoch {ep}")
        train_loss = []
        for inp1, inp2, inp3, lab in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            inputs = {'input_ids' : inp1.to(device), 'token_type_ids' : inp2.to(device), 'attention_mask' : inp3.to(device)}
            out = model(inputs)
            t_loss = loss_function(out, lab.to(device))
            t_loss.backward()
            optimizer.step()
            train_loss.append(t_loss.cpu().item())
        print(f"For learning rate {lrate}, the average training batch loss: {np.mean(train_loss)}")

        dev_accuracy_score_list = []
        for inp1, inp2, inp3, lab in tqdm(dev_loader):
            model.eval()
            targets=[]
            outputs=[]
            with torch.no_grad():
                inputs = {'input_ids' : inp1.to(device), 'token_type_ids' : inp2.to(device), 'attention_mask' : inp3.to(device)}
                out = model(inputs)
                targets.extend(lab.cpu().detach().numpy().tolist())
                outputs.extend(torch.argmax(torch.softmax(out, dim = 1), dim = 1).cpu().detach().numpy().tolist())
                d_acc_score = metrics.accuracy_score(targets, outputs)
                dev_accuracy_score_list.append(d_acc_score)
        print(f"For learning rate {lrate}, the development set accuracy score: {np.mean(dev_accuracy_score_list)}\n")
        if np.mean(dev_accuracy_score_list) > best_perf_dict["metric"]:
            best_perf_dict["metric"] = np.mean(dev_accuracy_score_list)
            best_perf_dict["epoch"]  = ep
            best_perf_dict["learning_rate"]  = lrate
            torch.save({
                "model_param": model.state_dict(),
                "optim_param": optimizer.state_dict(),
                "dev_metric": np.mean(dev_accuracy_score_list),
                "lr_metric": lrate,
                "epoch": ep
            }, f"/scratch/general/vast/u1405749/cs6957/assignment4/models/lr_{lrate}/model_{ep}")

print(f"""\nThe highest development set accuracy of {best_perf_dict["metric"]} is with learning rate {best_perf_dict["learning_rate"]}.""")

lr_best = best_perf_dict["learning_rate"]
epoch_best = best_perf_dict["epoch"]
model_path = f"/scratch/general/vast/u1405749/cs6957/assignment4/models/lr_{lr_best}/model_{epoch_best}"
checkpoint = torch.load(model_path)

model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])

print(f"""Development accuracy of loaded model: {checkpoint["dev_metric"]} with learning rate {checkpoint["lr_metric"]}""")

test_accuracy_score_list = []
random_classifier_accuracy_score_list = []
for inp1, inp2, inp3, lab in tqdm(test_loader):
    model.eval()
    targets=[]
    outputs=[]
    rand_outputs=[]
    with torch.no_grad():
        inputs = {'input_ids' : inp1.to(device), 'token_type_ids' : inp2.to(device), 'attention_mask' : inp3.to(device)}
        out = model(inputs)
        targets.extend(lab.cpu().detach().numpy().tolist())
        outputs.extend(torch.argmax(torch.softmax(out, dim = 1), dim = 1).cpu().detach().numpy().tolist())
        rand_outputs.extend(list(np.random.randint(2, size=len(lab.cpu().detach().numpy().tolist()))))
        t_acc_score = metrics.accuracy_score(targets, outputs)
        rand_acc_score = metrics.accuracy_score(targets, rand_outputs)
        test_accuracy_score_list.append(t_acc_score)
        random_classifier_accuracy_score_list.append(rand_acc_score)
print(f"The test set accuracy score: {np.mean(test_accuracy_score_list)}\n")
print(f"The test set accuracy of a random classifier: {np.mean(random_classifier_accuracy_score_list)}\n")

data_sst2 =  pd.read_csv('hidden_sst2.csv')
text_values = data_sst2["text"].tolist()
tokenized_data_sst2 = tokenizer(text_values, truncation=True, max_length = 512, padding='max_length', return_tensors='pt')
input_ids_sst2 = tokenized_data_sst2['input_ids']
token_type_ids_sst2 = tokenized_data_sst2['token_type_ids']
attention_masks_sst2 = tokenized_data_sst2['attention_mask']
dataset_sst2 = TensorDataset(input_ids_sst2, token_type_ids_sst2, attention_masks_sst2)
data_loader_sst2 = DataLoader(dataset_sst2, batch_size=1, shuffle=False)

pred_outputs = []
probab_0_outputs = []
probab_1_outputs = []
for inp1, inp2, inp3 in tqdm(data_loader_sst2):
    model.eval()
    with torch.no_grad():
        inputs = {'input_ids' : inp1.to(device), 'token_type_ids' : inp2.to(device), 'attention_mask' : inp3.to(device)}
        out = model(inputs)
        pred_outputs.extend(torch.argmax(torch.softmax(out, dim = 1), dim = 1).cpu().detach().numpy().tolist())
        probab_0_outputs.append(torch.softmax(out, dim = 1)[0][0].cpu().detach().numpy().tolist())
        probab_1_outputs.append(torch.softmax(out, dim = 1)[0][1].cpu().detach().numpy().tolist())
data_sst2['prediction'] = pred_outputs
data_sst2['probab_0'] = probab_0_outputs
data_sst2['probab_1'] = probab_1_outputs

data_sst2.to_csv('results_sst2.csv', index=False)

given_sentences = ["Kate should get promoted, she is an amazing employee.", "Bob should get promoted, he is an amazing employee.", "Kate should get promoted, he is an amazing employee.", "Bob should get promoted, they are an amazing employee."]
data_tokenized_sst2 = tokenizer(given_sentences, truncation=True, max_length = 512, padding='max_length', return_tensors='pt')
input_idss_sst2 = data_tokenized_sst2['input_ids']
token_type_idss_sst2 = data_tokenized_sst2['token_type_ids']
attention_maskss_sst2 = data_tokenized_sst2['attention_mask']
dataset_s_sst2 = TensorDataset(input_idss_sst2, token_type_idss_sst2, attention_maskss_sst2)
data_loader_s_sst2 = DataLoader(dataset_s_sst2, batch_size=1, shuffle=False)

preds_outputs = []
for inp1, inp2, inp3 in tqdm(data_loader_s_sst2):
    model.eval()
    with torch.no_grad():
        inputs = {'input_ids' : inp1.to(device), 'token_type_ids' : inp2.to(device), 'attention_mask' : inp3.to(device)}
        out = model(inputs)
        preds_outputs.extend(torch.argmax(torch.softmax(out, dim = 1), dim = 1).cpu().detach().numpy().tolist())
print(preds_outputs)      
