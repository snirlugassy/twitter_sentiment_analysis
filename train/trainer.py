from datetime import datetime
import os
import csv
import argparse
import torch

from models.gru import SentimentGRU_A
from models.transformer import SentimentTransformerEncoder_C

from dataset import SentimentAnalysisDataset
from test import run_test

RESULTS_DIR = 'results'

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adadelta': torch.optim.Adadelta,
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad
}

MODELS = {
    'a': SentimentGRU_A,
    'b': None,
    'c': SentimentTransformerEncoder_C
}

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train mask detection nerual network')
    argparser.add_argument('--data-path', type=str, required=True, dest='data_path')
    argparser.add_argument('--model', type=str, dest='model', default='a', choices=['a','b','c'])
    argparser.add_argument('--batch-size', type=int, dest='batch_size', default=256)
    argparser.add_argument('--epochs', type=int, dest='epochs', default=30)
    argparser.add_argument('--optimizer', type=str, dest='optimizer', choices=OPTIMIZERS.keys(), default='adam')
    argparser.add_argument('--lr', type=float, dest='lr', default=1e-3)
    argparser.add_argument('--print-steps', type=int, dest='print_steps', default=1000)
    args = argparser.parse_args()

    if args.optimizer not in OPTIMIZERS:
        args.optimizer = 'adam'

    data_path = args.data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('====== TRAIN =======')
    print('model:', args.model)
    print('optimizer:', args.optimizer)
    print('batch-size:', args.batch_size)
    print('epochs:', args.epochs)
    print('l-rate:', args.lr)
    print('device:', device)
    print('====================')

    print('-> Loading datasets')
    train_dataset = SentimentAnalysisDataset(os.path.join(data_path, 'trainEmotions.csv'))
    train_size = len(train_dataset)

    test_dataset = SentimentAnalysisDataset(os.path.join(data_path, 'testEmotions.csv'))
    test_size = len(test_dataset)

    print('-> Initalizing model')
    model = MODELS[args.model](100)
    model.to(device)
    model = model.double()
    print(f'Using model {model.__class__.__name__}')

    loss = torch.nn.CrossEntropyLoss()
    optimizer = OPTIMIZERS[args.optimizer](model.parameters(), lr=args.lr)

    t = datetime.now().strftime('%m_%d_%H_%M')

    print('-> Running pre-training test')
    test_acc, _, test_loss = run_test(model, test_dataset, loss, device)
    print('Accuracy:', test_acc)
    print('Loss:', test_loss)
    print('---------------')

    results = []
    max_test_acc = 0
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
                output = model(tokens)
                assert output.dim() == 1 and output.shape[0] == 3
                L = loss(output, label.argmax())
                train_loss += L.item()
                L.backward()
            except Exception as e:
                print(i, tokens, label)
                print(e)

            if i & args.batch_size == 0:
                # Batched Backpropagation
                optimizer.step()
                optimizer.zero_grad()

            if i % args.print_steps == 0:
                print(f'L: {train_loss / (i+1):>7f}  [{i}/{train_size}]')

        print(f'Epoch Loss: {train_loss / train_size}\n------------------')

        print('-> Running test')
        test_acc, _, test_loss = run_test(model, test_dataset, loss, device)
        print('Accuracy:', test_acc)
        print('Loss:', test_loss)
        print('---------------')

        if test_acc > max_test_acc:    
            print('-> Saving state')
            torch.save(model.state_dict(), f'models/{args.model_name.upper()}_{t}.state')

        results.append({
            'epoch': epoch,
            'train_loss': train_loss / train_size,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
        })

    print('Saving results CSV')
    with open(f'results/{args.model_name.upper()}_{t}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result)
