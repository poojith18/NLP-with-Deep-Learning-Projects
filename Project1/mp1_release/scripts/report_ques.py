import numpy as np

embeddings_file = 'embeddings.txt'
word_embeddings = {}
with open(embeddings_file, 'r') as file:
    lines = file.readlines()[1:]
    for line in lines:
        parts = line.strip().split()
        word = parts[0]
        embedding = np.array([float(val) for val in parts[1:]])
        word_embeddings[word] = embedding

def cosine_similarity(embedding1, embedding2):
    product = np.dot(embedding1, embedding2)
    norm_1 = np.linalg.norm(embedding1)
    norm_2 = np.linalg.norm(embedding2)
    cos_similarity = product / (norm_1 * norm_2)
    return cos_similarity

pairs = [
    ("cat", "tiger"),
    ("plane", "human"),
    ("my", "mine"),
    ("happy", "human"),
    ("happy", "cat"),
    ("king", "princess"),
    ("ball", "racket"),
    ("good", "ugly"),
    ("cat", "racket"),
    ("good", "bad")
]

for w1, w2 in pairs:
    sim = cosine_similarity(word_embeddings[w1], word_embeddings[w2])
    print(f"Similarity score for '{w1}' and '{w2}': {sim:.4f}.")
print()

def most_similar_word(embedding_vector, word_embeddings):
    max_sim = -1
    most_sim = None
    for word, embedding in word_embeddings.items():
        sim = cosine_similarity(embedding_vector, embedding)
        if sim > max_sim:
            max_sim = sim
            most_sim = word
    return most_sim

analogies = [
    ("king", "queen", "man"),
    ("king", "queen", "prince"),
    ("king", "man", "queen"),
    ("woman", "man", "princess"),
    ("prince", "princess", "man")
]

for wa, wb, wc in analogies:
    res_vec = word_embeddings[wb] - word_embeddings[wa] + word_embeddings[wc]
    res_word = most_similar_word(res_vec, word_embeddings)
    print(f"{wa}:{wb}, {wc}:{res_word}")
