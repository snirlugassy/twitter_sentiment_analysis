import torch
import numpy as np
from dataset import LABEL_MAP
from sklearn.metrics import accuracy_score, confusion_matrix

def run_test(model, dataset, loss_func, device):

    test_loss = 0.0
    y_true = []
    y_predict = []

    model.eval()
    with torch.no_grad():
        for tokens, label in dataset:
            tokens = tokens.to(device).unsqueeze(0)
            label = label.to(device).unsqueeze(0)
            y_true.append(int(label.argmax(dim=1)))

            if tokens.squeeze().dim() == 0 or len(tokens.squeeze()) == 0:
                # Predict neutral if no token after processing 
                # e.g., only stopwords in the original text
                y_predict.append(int(LABEL_MAP['neutral']))
                continue

            # Forward pass
            output = model(tokens)
            y_prob = output.squeeze(0)[-1]
            y_predict.append(int(torch.softmax(y_prob, dim=0).argmax(dim=0)))
            L = loss_func(y_prob.view(1,-1), label.view(1,-1))
            test_loss += L.item()

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    return accuracy_score(y_true, y_predict), confusion_matrix(y_true, y_predict), test_loss