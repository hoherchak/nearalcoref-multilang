from shutil import copyfile

import numpy as np
from navec import Navec


path = 'ru/navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)
words = []
embeddings = []
for word, id in navec.vocab.word_ids.items():
    if word == '<unk>':
        word = '*UNK*'
    words.append(word)
    embeddings.append(navec.pq[id])
embeddings = np.array(embeddings).astype(np.float)

np.save('ru/static_word_embeddings.npy', embeddings)
with open('ru/static_word_vocabulary.txt', 'w') as f:
    for word in words:
        f.write("%s\n" % word)

copyfile('ru/static_word_embeddings.npy', 'ru/tuned_word_embeddings.npy')
copyfile('ru/static_word_vocabulary.txt', 'ru/tuned_word_vocabulary.txt')
