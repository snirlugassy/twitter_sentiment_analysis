import os
import sys
import torch
import numpy as np
from dataset import LABEL_MAP
from sklearn.metrics import accuracy_score, confusion_matrix

from dataset import SentimentAnalysisDataset

import argparse
import torch

from model import TransformerModel, SentimentGRUWithGlove
from dataset import SentimentAnalysisDataset

def run_test(model, dataset, loss_func, device):

    test_loss = 0.0
    y_true = []
    y_predict = []

    model.eval()
    with torch.no_grad():
        for tokens, label in dataset:
            tokens = tokens.to(device).unsqueeze(0)
            label = label.to(device)
            y_true.append(int(label.argmax()))

            if tokens.squeeze().dim() == 0 or len(tokens.squeeze()) == 0:
                # Predict neutral if no token after processing 
                # e.g., only stopwords in the original text
                y_predict.append(int(LABEL_MAP['neutral']))
                continue

            # Forward pass
            output = model(tokens)
            y_prob = output.squeeze(0)
            y_predict.append(int(torch.softmax(y_prob, dim=0).argmax(dim=0)))
            L = loss_func(y_prob.view(1,-1), label.view(1,-1))
            test_loss += L.item()

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    return accuracy_score(y_true, y_predict), confusion_matrix(y_true, y_predict), test_loss

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Test mask detection nerual network')
    argparser.add_argument('--data-path', type=str, required=True, dest='data_path')
    argparser.add_argument('--model', type=str, required=True, dest='model', choices=['gru', 'transformer'])
    argparser.add_argument('--model-path', type=str, required=True, dest='model_path')
    args = argparser.parse_args()

    test_dataset = SentimentAnalysisDataset(args.data_path)
    test_size = len(test_dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'gru':
        model = SentimentGRUWithGlove(100)
    else:
        model = TransformerModel(100)
    
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()

    acc, conf_mat = run_test(model, test_dataset, loss, device)
    print('accuracy:', acc)
    print('confusion matrix:', conf_mat)