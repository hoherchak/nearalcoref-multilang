from shutil import copyfile

import numpy as np
from tqdm import tqdm


def load_glove(file):
    """Loads GloVe vectors in numpy array.
    Args:
        file (str): a path to a glove file.
    Return:
        dict: a dict of numpy arrays.
    """
    embeddings = []
    words = []
    with open(file, encoding='utf8') as f:
        for i, line in tqdm(enumerate(f)):
            values = line.split()
            word = ''.join(values[:-300])
            if word == '<unk>':
                word = '*UNK*'
            coefs = values[-300:]
            embeddings.append(coefs)
            words.append(word)

    return np.array(embeddings).astype(np.float), words


# EMBEDDING_PATH = '../embedding_weights/glove.840B.300d.txt'
EMBEDDING_PATH = 'fiction.lowercased.lemmatized.glove.300d'
embeddings, words = load_glove(EMBEDDING_PATH)

np.save('static_word_embeddings.npy', embeddings)
with open('static_word_vocabulary.txt', 'w') as f:
    for word in words:
        f.write("%s\n" % word)

copyfile('static_word_embeddings.npy', 'tuned_word_embeddings.npy')
copyfile('static_word_vocabulary.txt', 'tuned_word_vocabulary.txt')
