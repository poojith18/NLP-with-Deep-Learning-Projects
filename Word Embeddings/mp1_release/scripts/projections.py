import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

embeddings_file = 'embeddings.txt'
word_embeddings = {}
with open(embeddings_file, 'r') as file:
    lines = file.readlines()[1:]
    for line in lines:
        parts = line.strip().split()
        word = parts[0]
        embedding = np.array([float(v) for v in parts[1:]])
        word_embeddings[word] = embedding

words_to_project = [
    "horse", "cat", "dog", "i", "he", "she", "it", "her", "his", "our", "we", "in", "on",
    "from", "to", "at", "by", "man", "woman", "boy", "girl", "king", "queen", "prince", "princess"
]

word_vectors = [word_embeddings[word] for word in words_to_project]

pca = PCA(n_components=2)
r_word_vectors = pca.fit_transform(word_vectors)

plt.figure(figsize=(10, 8))
plt.scatter(r_word_vectors[:, 0], r_word_vectors[:, 1], marker='o', color='b', s=10)
for i, word in enumerate(words_to_project):
    plt.annotate(word, (r_word_vectors[i, 0], r_word_vectors[i, 1]))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2-D Projection')
plt.grid(True)
plt.show()
plt.savefig('projections.png')
