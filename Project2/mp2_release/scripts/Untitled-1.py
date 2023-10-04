def predict_actions(model, tokens, pos_tags):
    model.eval()
    with torch.no_grad():
        output = model(tokens, pos_tags)
    predicted_actions = torch.argmax(output, dim=1)
    return predicted_actions

def predict_actions_probs(model, tokens, pos_tags):
    model.eval()
    with torch.no_grad():
        output = model(tokens, pos_tags)
    return output

def dev_test_data_preprocess(file_data):
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
            pred_act_id = predict_actions(parser,mean_token_embs,pos_tokens_ix)
            pred_actions = predict_actions_probs(parser,mean_token_embs,pos_tokens_ix)
            act = tagset_ix2word[pred_act_id]
            if len(t_parser.stack) <= 3
                act = "SHIFT":
            elif len(t_parser.parse_buffer)<=2:
                pred_actions.pop(0)
                prediction_act_id = torch.argmax(pred_actions, dim=1)
                act = tagset_ix2word[prediction_act_id]
            if act == "SHIFT":
                state.shift(t_parser)
            elif act.startswith("REDUCE_L"):
                state.left_arc(t_parser, act.split("_")[2])
            else:
                state.right_arc(t_parser, act.split("_")[2])
            act_list.append(act)
        pred_action_list.append(act_list)    
    return pred_action_list

with open('results.txt', 'w') as f:
    for inp1,inp2 in tqdm(hidden_loader):
        predicted_actions = predict_actions(parser, inp1, inp2)
        for word, idx in tagset_word2ix.items():
            action_strings = [tagset_ix2word[action] for action in predicted_actions]
        action_string = ' '.join(action_strings)
        f.write(action_string + '\n')






device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
max_epochs = 2
for ep in range(1, max_epochs+1):
    print(f"Epoch {ep}")
    train_loss = []
    for inp1,inp2,lab in tqdm(train_loader):
        model.train()
        optimizer.zero_grad()
        out = model(inp1.to(device),inp2.to(device))
        t_loss = loss_function(out, lab.to(device))
        t_loss.backward()
        optimizer.step()
        train_loss.append(t_loss.cpu().item())
    print(f"For learning rate {lrate}, the average training batch loss: {np.mean(train_loss)}")

for inp1,inp2,lab in tqdm(train_loader):
    parser.train()
    optimizer.zero_grad()
    out = parser(inp1,inp2)
    t_loss = loss_function(out, lab)
    t_loss.backward()
    optimizer.step()
    print(t_loss.cpu().item())
    
len(train_read_file_data)
c = 2
demb_values = [50, 300]
dpos = 50
h = 200
num_actions = 75
learning_rates = [0.01, 0.001, 0.0001]
max_epochs = 20

glove_embeddings = vocab.GloVe(name='6B', dim=300)

class TransitionParser(nn.Module):
    def __init__(self, vocab_size, pos_vocab_size, demb, dpos, h, num_actions):
        super(TransitionParser, self).__init__()
        self.word_embedding = nn.Linear(demb, h)
        self.pos_embedding = nn.Embedding(dpos, h)
        self.fc1 = nn.Linear(2 * c * (demb + dpos), h)
        self.fc2 = nn.Linear(h, num_actions)

    def forward(self, stack, buffer, pos_stack, pos_buffer):
        word_embeds = self.word_embedding(torch.cat([stack, buffer], dim=0))
        pos_embeds = self.pos_embedding(torch.cat([pos_stack, pos_buffer], dim=0))
        word_mean = torch.mean(word_embeds, dim=0)
        pos_mean = torch.mean(pos_embeds, dim=0)
        combined_representation = torch.cat([word_mean, pos_mean], dim=0)
        hidden_rep = torch.relu(self.fc1(combined_representation))
        action_probs = torch.softmax(self.fc2(hidden_rep), dim=0)
        return action_probs

best_dev_las = -1
best_model = None
best_lr = None

for lr in learning_rates:
    parser = TransitionParser(demb=300)
    optimizer = optim.Adam(parser.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(max_epochs):
        random.seed(42)  # Fix the random seed for reproducibility
        for batch in training_data:
            # Forward pass: compute predicted actions using parser.forward
            # Compute loss using criterion
            # Backpropagation and optimization step using optimizer

    # Evaluate your model on the development set to select the best model based on LAS
    dev_las = evaluate_model(parser, dev_data)  # Implement this function based on your evaluation method
    if dev_las > best_dev_las:
        best_dev_las = dev_las
        best_model = parser
        best_lr = lr

# Test your best model on the test set and report UAS and LAS
test_uas, test_las = evaluate_model(best_model, test_data)  # Implement this function based on your evaluation method

# Save your predicted actions for the hidden data to results.txt
with open('results.txt', 'w') as f:
    for sentence in hidden_data:
        predicted_actions = predict_actions(best_model, sentence)  # Implement this function to predict actions
        f.write(" ".join(predicted_actions) + "\n")

# Implement the extra credit task if you choose to do it

# Report your findings based on the project requirements
print("Best Dev LAS:", best_dev_las)
print("Best Learning Rate:", best_lr)
print("Test UAS:", test_uas)
print("Test LAS:", test_las)