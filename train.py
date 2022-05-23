import os
import csv
import argparse
import pickle
import torch

from model import SentimentLSTM
from dataset import SentimentAnalysisDataset
from test import run_test

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adadelta': torch.optim.Adadelta,
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad
}

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train mask detection nerual network')
    argparser.add_argument('--data-path', type=str, required=True, dest='data_path')
    argparser.add_argument('--batch-size', type=int, dest='batch_size', default=64)
    argparser.add_argument('--epochs', type=int, dest='epochs', default=30)
    argparser.add_argument('--optimizer', type=str, dest='optimizer', choices=OPTIMIZERS.keys(), default='adam')
    argparser.add_argument('--lr', type=float, dest='lr', default=1e-3)
    argparser.add_argument('--print-steps', type=int, dest='print_steps', default=200)
    args = argparser.parse_args()

    if args.optimizer not in OPTIMIZERS:
        args.optimizer = 'adam'

    data_path = args.data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('====== TRAIN =======')
    print('optimizer:', args.optimizer)
    print('batch-size:', args.batch_size)
    print('epochs:', args.epochs)
    print('l-rate:', args.lr)
    print('device:', device)
    print('====================')

    print('-> Loading vocabulary')
    with open('vocab.text', 'rb') as pkl:
        vocab = pickle.load(pkl)

    print('-> Loading datasets')
    train_dataset = SentimentAnalysisDataset(os.path.join(data_path, 'trainEmotions.csv'), vocab)
    train_size = len(train_dataset)

    test_dataset = SentimentAnalysisDataset(os.path.join(data_path, 'testEmotions.csv'), vocab)
    test_size = len(test_dataset)

    print('-> Initalizing model')
    model = SentimentLSTM(len(vocab))
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = OPTIMIZERS[args.optimizer](model.parameters(), lr=args.lr)

    output = []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}\n---------------------------")

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
                y_prob = model(tokens)[-1]
            except Exception as e:
                print(i,tokens, label)
                print(e)
            L = loss(y_prob.view(1,-1), label.view(1,-1))
            train_loss += L.item()
            L.backward()

            if i & args.batch_size == 0:
                # Batched Backpropagation
                optimizer.step()
                optimizer.zero_grad()

            if i % args.print_steps == 0:
                print(f'L: {train_loss / (i+1):>7f}  [{i}/{train_size}]')

        print(f'Epoch Loss: {train_loss / train_size}\n------------------')

        print('-> Running test')
        f1, acc, conf_mat, test_loss = run_test(model, test_dataset, loss, device)
        print('F1-Score:', f1)
        print('Accuracy:', acc)
        print('Loss:', test_loss)
        print('---------------')

        print('-> Saving state')
        torch.save(model.state_dict(), 'model.state')

    # print('Saving output CSV')
    # with open('output.csv', 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=list(output[0].keys()))
    #     writer.writeheader()
    #     for x in output:
    #         writer.writerow(x)
