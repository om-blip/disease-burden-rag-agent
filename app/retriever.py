import numpy as np
import faiss

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


class HybridRetriever:

    def __init__(self, chunks):

        self.chunks = chunks

        # -------------------------
        # TF-IDF RETRIEVER
        # -------------------------

        self.vectorizer = TfidfVectorizer()

        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)

        # -------------------------
        # EMBEDDING MODEL
        # -------------------------

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # -------------------------
        # CREATE EMBEDDINGS
        # -------------------------

        embeddings = self.model.encode(chunks)

        self.embeddings = np.array(embeddings).astype("float32")

        # -------------------------
        # BUILD FAISS INDEX
        # -------------------------

        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)

        self.index.add(self.embeddings)

    # -------------------------
    # SEARCH FUNCTION
    # -------------------------

    def search(self, query, k=5):

        # -------------------------
        # TF-IDF SEARCH
        # -------------------------

        query_vec = self.vectorizer.transform([query])

        tfidf_scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()

        tfidf_idx = np.argsort(tfidf_scores)[-k:][::-1]

        # -------------------------
        # FAISS VECTOR SEARCH
        # -------------------------

        query_embedding = self.model.encode([query])

        query_embedding = np.array(query_embedding).astype("float32")

        distances, faiss_idx = self.index.search(query_embedding, k)

        faiss_idx = faiss_idx[0]

        # -------------------------
        # MERGE RESULTS
        # -------------------------

        combined_idx = list(set(tfidf_idx.tolist() + faiss_idx.tolist()))

        results = [self.chunks[i] for i in combined_idx]

        scores = []

        for i in combined_idx:
            score = tfidf_scores[i]
            scores.append(score)

        scores = np.array(scores)

        return results, scores