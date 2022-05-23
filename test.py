import torch
import numpy as np
from dataset import LABEL_MAP
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def run_test(model, dataset, loss_func, device):

    test_loss = 0.0
    y_true = []
    y_predict = []

    model.eval()
    with torch.no_grad():
        for tokens, label in dataset:
            tokens = tokens.to(device)
            label = label.to(device)
            y_true.append(int(label.argmax(dim=0)))

            if len(tokens) == 0:
                # Predict neutral if no token after processing 
                # e.g., only stopwords in the original text
                y_predict.append(int(LABEL_MAP['neutral']))
                continue

            # Forward pass
            y_prob = model(tokens)[-1]
            y_predict.append(int(torch.softmax(y_prob, dim=0).argmax(dim=0)))
            L = loss_func(y_prob.view(1,-1), label.view(1,-1))
            test_loss += L.item()

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    return f1_score(y_true, y_predict, average='micro'), accuracy_score(y_true, y_predict), confusion_matrix(y_true, y_predict), test_loss