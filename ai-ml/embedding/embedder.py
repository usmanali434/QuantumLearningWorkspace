"""
Orchestrates the full embedding pipeline:

  document (common ingestion schema)
      -> chunk (chunker.py)
      -> embed (model.py, free local sentence-transformers)
      -> store full chunk text + metadata in MongoDB   (source of truth)
      -> store vector + small pointer metadata in Pinecone (for search)

MongoDB holds the full text because Pinecone metadata has size/type
limits and isn't meant for large text blobs. Pinecone only stores the
vector plus a small pointer (chunk id + a few filterable fields) so a
similarity search can be resolved back to the full chunk via MongoDB.
"""

import uuid
from pymongo import MongoClient

from embedding.config import settings
from embedding.chunker import chunk_document
from embedding.model import get_embedding_model
from embedding.vector_store import PineconeVectorStore

import argparse
import requests


class Embedder:
    def __init__(self, model=None, vector_store=None, mongo_client=None):
        self.model = model or get_embedding_model()
        self.vector_store = vector_store or PineconeVectorStore(dimension=self.model.dimension())

        self._mongo_client = mongo_client or MongoClient(settings.mongodb_uri)
        self._db = self._mongo_client[settings.mongodb_db]
        self._collection = self._db[settings.mongodb_collection]

    def embed_document(self, document: dict) -> dict:
        """
        document: common ingestion schema
          { "source_type": ..., "title": ..., "text": ..., "metadata": {...} }

        Chunks it, embeds every chunk, saves full text to MongoDB, and
        upserts vectors to Pinecone. Returns a small summary dict.
        """
        chunks = chunk_document(document)
        if not chunks:
            return {"document_id": None, "chunks_stored": 0}

        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(texts)

        document_id = str(uuid.uuid4())
        mongo_docs = []
        pinecone_vectors = []

        for chunk, vector in zip(chunks, vectors):
            chunk_id = f"{document_id}_{chunk['chunk_index']}"

            mongo_docs.append({
                "_id": chunk_id,
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "title": chunk["title"],
                "source_type": chunk["source_type"],
                "metadata": chunk["metadata"],
            })

            pinecone_vectors.append({
                "id": chunk_id,
                "values": vector,
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": chunk["chunk_index"],
                    "title": chunk["title"],
                    "source_type": chunk["source_type"],
                },
            })

        if mongo_docs:
            self._collection.insert_many(mongo_docs)

        self.vector_store.upsert(pinecone_vectors)

        return {"document_id": document_id, "chunks_stored": len(mongo_docs)}

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Embeds the query, finds the nearest chunks in Pinecone, then
        resolves each match back to its full text stored in MongoDB.
        """
        query_vector = self.model.encode(query)[0]
        matches = self.vector_store.query(query_vector, top_k=top_k)

        results = []
        for match in matches:
            chunk_id = match["id"] if isinstance(match, dict) else match.id
            score = match["score"] if isinstance(match, dict) else match.score

            mongo_doc = self._collection.find_one({"_id": chunk_id})
            if mongo_doc:
                results.append({
                    "score": score,
                    "text": mongo_doc["text"],
                    "title": mongo_doc["title"],
                    "source_type": mongo_doc["source_type"],
                    "metadata": mongo_doc["metadata"],
                })
        return results

    def delete_document(self, document_id: str, chunk_count: int):
        """Remove a document's chunks from both MongoDB and Pinecone."""
        ids = [f"{document_id}_{i}" for i in range(chunk_count)]
        self._collection.delete_many({"document_id": document_id})
        self.vector_store.delete(ids)

    def close(self):
        self._mongo_client.close()




INGESTION_BASE_URL = "http://127.0.0.1:8000"


def _fetch_from_ingestion(pdf=None, youtube=None, article=None) -> dict:
    if pdf:
        with open(pdf, "rb") as f:
            response = requests.post(f"{INGESTION_BASE_URL}/ingest/pdf", files={"file": f})
    elif youtube:
        response = requests.post(f"{INGESTION_BASE_URL}/ingest/youtube", json={"url": youtube})
    elif article:
        response = requests.post(f"{INGESTION_BASE_URL}/ingest/article", json={"url": article})
    else:
        raise ValueError("Provide one of pdf, youtube, or article")

    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    # python -m embedding.embedder --pdf path/to/file.pdf
    # python -m embedding.embedder --youtube "https://youtube.com/watch?v=..."
    # python -m embedding.embedder --article "https://example.com/article"
    # python -m embedding.embedder                (no args -> runs built-in sample test)

    parser = argparse.ArgumentParser(description="Run ingestion -> embedding end to end")
    parser.add_argument("--pdf", help="Path to a local PDF file")
    parser.add_argument("--youtube", help="YouTube video URL")
    parser.add_argument("--article", help="Web article URL")
    args = parser.parse_args()

    if args.pdf or args.youtube or args.article:
        print("Calling ingestion...")
        document = _fetch_from_ingestion(pdf=args.pdf, youtube=args.youtube, article=args.article)
        print(f"Ingestion returned title: {document.get('title', '(no title)')!r}")
        print(f"Text length: {len(document.get('text', ''))} characters")
    else:
        document = {
            "source_type": "article",
            "title": "Intro to Machine Learning",
            "text": (
                "Machine learning is a branch of artificial intelligence. "
                "It focuses on building systems that learn from data. "
                "Supervised learning uses labeled data to train models."
            ),
            "metadata": {"author": "", "date": "", "source": "https://example.com"},
        }

    embedder = Embedder()
    summary = embedder.embed_document(document)
    print("Embedded:", summary)

    query = document.get("title") or document.get("text", "")[:50]
    results = embedder.search(query, top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['text'][:100]}...")

    embedder.close()