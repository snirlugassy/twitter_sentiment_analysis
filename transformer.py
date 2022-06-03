#!/usr/bin/env python
# coding: utf-8

# In[31]:


from datetime import datetime
import os
import csv
import argparse
import pickle
import torch

from model import TransformerModel, SentimentGRUWithGlove
from dataset import SentimentAnalysisDataset
from test import run_test

import numpy as np
from dataset import LABEL_MAP
from sklearn.metrics import accuracy_score, confusion_matrix


class TransformerModel(torch.nn.Module):
    def __init__(self, input_size=100, embedding_dim=200) -> None:
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_size, nhead=5, batch_first=True)
        self.tranformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.33),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,3)
        )

    def forward(self, x):
        x = self.tranformer(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

model = TransformerModel(input_size=100)


# In[6]:


OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adadelta': torch.optim.Adadelta,
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad
}

args = {
    'data_path': 'data',
    'model_name': 'transformer_classifier_C',
    'batch_size': 256,
    'epochs': 30,
    'optimizer': 'adam',
    'lr': 1e-3,
    'print_steps': 1000
}

data_path = args['data_path']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('-> Loading datasets')
train_dataset = SentimentAnalysisDataset(os.path.join(data_path, 'trainEmotions.csv'))
train_size = len(train_dataset)

test_dataset = SentimentAnalysisDataset(os.path.join(data_path, 'testEmotions.csv'))
test_size = len(test_dataset)


# In[53]:


print('-> Initalizing model')
model = TransformerModel(input_size=100)
model.to(device)
model = model.float()
print(f'Using model {model.__class__.__name__}')

loss = torch.nn.CrossEntropyLoss()
optimizer = OPTIMIZERS[args['optimizer']](model.parameters(), lr=args['lr'])

t = datetime.now().strftime('%m_%d_%H_%M')

results = []
max_test_acc = 0


# In[62]:


def run_test(model, dataset, loss_func, device):

    test_loss = 0.0
    y_true = []
    y_predict = []

    model.eval()
    with torch.no_grad():
        for tokens, label in dataset:
            
            tokens = tokens.to(device).float()
            label = label.to(device)

            y_true.append(int(label.argmax()))

            if tokens.squeeze().dim() == 0 or len(tokens.squeeze()) == 0:
                # Predict neutral if no token after processing 
                # e.g., only stopwords in the original text
                y_predict.append(int(LABEL_MAP['neutral']))
                continue

            # Forward pass
            output = model(tokens)
            y_predict.append(int(torch.softmax(output, dim=0).argmax()))
            L = loss_func(output.view(1,-1), label.view(1,-1))
            test_loss += L.item()

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    return accuracy_score(y_true, y_predict), confusion_matrix(y_true, y_predict), test_loss


# In[63]:


print('-> Running test')
test_acc, _, test_loss = run_test(model, test_dataset, loss, device)
print('Accuracy:', test_acc)
print('Loss:', test_loss)
print('---------------')


# In[30]:


for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1}/{args['epochs']}\n---------------------------")

    train_loss = 0.0
    model.train()
    optimizer.zero_grad()
    for i, (tokens, label) in enumerate(train_dataset):
        if len(tokens) == 0:
            continue

        tokens = tokens.to(device)
        label = label.to(device)

        # Forward pass
        try:
            output = model(tokens)
            L = loss(output.view(1,-1), label.view(1,-1))
            train_loss += L.item()
            L.backward()
        except Exception as e:
            print(i,tokens, label)
            print(e)

        if i & args['batch_size'] == 0:
            # Batched Backpropagation
            optimizer.step()
            optimizer.zero_grad()

        if i % args['print_steps'] == 0:
            print(f'L: {train_loss / (i+1):>7f}  [{i}/{train_size}]')

    print(f'Epoch Loss: {train_loss / train_size}\n------------------')

    print('-> Running test')
    test_acc, _, test_loss = run_test(model, test_dataset, loss, device)
    print('Accuracy:', test_acc)
    print('Loss:', test_loss)
    print('---------------')

    if test_acc > max_test_acc:    
        print('-> Saving state')
        torch.save(model.state_dict(), f'{args["model_name"]}_{t}.state')

    results.append({
        'epoch': epoch,
        'train_loss': train_loss / train_size,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
    })


# In[ ]:


print('Saving results CSV')
with open(f'results_{args['model_name']}_{t}.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
    writer.writeheader()
    for result in results:
        writer.writerow(result)

