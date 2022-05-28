import re
from nltk.stem import PorterStemmer
THREE_DOTS_TOKEN = '3NEKUDOT'

stemmer = PorterStemmer()

with open('stopwords.txt', 'r') as f:
    STOPWORDS = [x.strip() for x in f.readlines()]

def text_processing(s:str):
    # remove tags
    s = re.sub(r'@[^\s]+', ' ', s)
    # add space to ! or ?
    s = re.sub('[!]', ' ! ', s)
    s = re.sub('[?]', ' ? ', s)
    # replace 3+ docs with special token
    s = re.sub(r'\.{3,}', ' ' + THREE_DOTS_TOKEN + ' ', s)
    # remove URL
    s = re.sub(r'(?:(?:https?|ftp):\/\/)?[\w\/\-?=%.]+\.[\w\/\-&?=%.]+', ' ', s)
    # remove special characters
    s = re.sub(r'[^a-zA-Z\s!?]', ' ', s)
    # remove extra space with a single space
    s = re.sub(r'\s\s+', ' ', s)
    # convert to lower
    s = s.lower()
    # tokenize and remove stopwords
    tokens = [stemmer.stem(w) for w in s.split(' ') if w not in STOPWORDS]
    return tokens
