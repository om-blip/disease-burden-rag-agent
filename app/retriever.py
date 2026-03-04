import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss

class HybridRetriever:

    def __init__(self, docs):

        self.docs = docs

        # TF-IDF
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(docs)

        # Embeddings
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.embed_model.encode(docs)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

        self.embeddings = embeddings

    def search(self, query, k=10):

        # TF-IDF scores
        q_vec = self.tfidf.transform([query])
        tfidf_scores = (self.tfidf_matrix @ q_vec.T).toarray().flatten()

        # Embedding search
        q_emb = self.embed_model.encode([query])
        distances, idx = self.index.search(np.array(q_emb), k)

        emb_scores = np.zeros(len(self.docs))
        emb_scores[idx[0]] = 1 / (1 + distances[0])

        # Combine scores
        hybrid_scores = tfidf_scores + emb_scores

        top_idx = np.argsort(hybrid_scores)[-k:][::-1]

        results = [self.docs[i] for i in top_idx]

        return results, hybrid_scores[top_idx]