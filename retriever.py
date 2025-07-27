from embedder import MODEL_NAME, INDEX, QUESTIONS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# FAQ entries — should eventually move to a persistent JSON/YAML file
FAQ_ENTRIES = [
    {"question": "What are your working hours?", "answer": "We are open from 9 AM to 6 PM, Monday to Friday."},
    {"question": "Where are you located?", "answer": "Our office is at 123 Main Street, Springfield."},
    {"question": "How can I reset my password?", "answer": "Click on 'Forgot Password' at login and follow instructions."},
    {"question": "Do you offer customer support?", "answer": "Yes, we offer 24/7 customer support via chat and email."},
]


class Retriever:
    """
    A retrieval-augmented class for matching user queries to a set of predefined FAQ entries
    using dense semantic search with FAISS and a SentenceTransformer model.

    Attributes:
        model (SentenceTransformer): Pretrained sentence embedding model.
        index (faiss.Index): FAISS index storing vectorized FAQ questions.
        questions (list[str]): List of original question texts with 'passage:' prefix used for embedding.
        faq_data (list[dict]): List of FAQ entries with 'question' and 'answer' keys.

    Methods:
        search(user_query: str, top_k: int = 1) -> list[dict]:
            Encodes the user query, performs FAISS similarity search,
            and returns the top-k most relevant FAQ question-answer pairs.
    """
    def __init__(self, model_name: str, index_path: str, questions_path: str, faq_data: list[dict]):
        """
        Load the embedding model, FAISS index, and pre-encoded questions.
        """
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.questions = np.load(questions_path, allow_pickle=True).tolist()
        self.faq_data = faq_data

    def search(self, user_query: str, top_k: int = 1) -> list[dict]:
        """
        Given a user query, retrieve the top-k most relevant FAQ answers.
        """
        encoded_query = self.model.encode([f"query: {user_query}"], normalize_embeddings=True) # query prefix required by the model
        distances, indices = self.index.search(np.array(encoded_query), top_k)

        results = []
        for idx in indices[0]:
            faq_entry = self.faq_data[idx]
            results.append({
                "question": faq_entry["question"],
                "answer": faq_entry["answer"]
            })
        return results


# if __name__ == "__main__":
#     retriever = Retriever(
#         model_name=MODEL_NAME,
#         index_path=INDEX,
#         questions_path=QUESTIONS,
#         faq_data=FAQ_ENTRIES
#     )

#     while "exit" not in (user_input := input("Ask a question (type 'exit' to quit): ")):
#         top_answers = retriever.search(user_input)
#         for result in top_answers:
#             print(f"\nBest Match: {result['question']}\nAnswer: {result['answer']}")
