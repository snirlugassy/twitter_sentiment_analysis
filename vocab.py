
import pickle
import pandas as pd
from processing import text_processing


if __name__ == '__main__':
    print('Loading training data')
    data = pd.read_csv('data/trainEmotions.csv')

    print('Processing text')
    data['clean'] = data['content'].apply(text_processing)

    print('Building vocabulary')
    vocab = {}
    next_idx = 0
    for text in data['clean']:
        for word in text:
            if word not in vocab.keys():
                vocab[word] = next_idx
                next_idx += 1

    print('Saving to file')
    with open('vocab.dict', 'wb') as pkl:
        pickle.dump(vocab, pkl) 