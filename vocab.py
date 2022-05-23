
import pickle
import pandas as pd
from torchtext import vocab
from processing import text_processing


if __name__ == '_main__':
    print('Loading training data')
    data = pd.read_csv('data/trainEmotions.csv')

    print('Processing text')
    data['clean'] = data['content'].apply(text_processing)

    print('Building vocabulary')
    voc = vocab.build_vocab_from_iterator(data['clean'], min_freq=1)

    print('Saving to file')
    with open('vocab.text', 'wb') as pkl:
        pickle.dump(voc, pkl) 