import sys
import os
import torch

import csv
import argparse

from models.transformer import SentimentTransformerEncoder_C

from dataset import SentimentAnalysisDataset, LABEL_MAP, LABEL_MAP_INV


if __name__ == '__main__':
    data_path = sys.argv[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentTransformerEncoder_C()
    model.load_state_dict(torch.load('model.pkl', map_location=device))
    model = model.float()
    dataset = SentimentAnalysisDataset(data_path, shuffle=False)
    
    y_predict = []
    model.eval()
    with torch.no_grad():
        for tokens, label in dataset:
            
            tokens = tokens.to(device).float()
            label = label.to(device)

            if tokens.squeeze().dim() == 0 or len(tokens.squeeze()) == 0:
                # Predict neutral if no token after processing 
                # e.g., only stopwords in the original text
                y_predict.append(int(LABEL_MAP['neutral']))
                continue

            # Forward pass
            output = model(tokens)
            y_predict.append(int(torch.softmax(output, dim=0).argmax()))

    with open('prediction.csv', 'w') as output_file:
        output_file.write('emotion,content\n')
        for i, y in enumerate(y_predict):
            output_file.write(f'{LABEL_MAP_INV[y]}, {dataset.df.iloc[i].content}\n')
