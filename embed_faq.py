from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MODEL_NAME = "intfloat/e5-base-v2"
INDEX = "faq.index"
QUESTIONS = "faq_questions.npy"
faq_entries = [
    {"question": "What are your working hours?", "answer": "We are open from 9 AM to 6 PM, Monday to Friday."},
    {"question": "Where are you located?", "answer": "Our office is at 123 Main Street, Springfield."},
    {"question": "How can I reset my password?", "answer": "Click on 'Forgot Password' at login and follow instructions."},
    {"question": "Do you offer customer support?", "answer": "Yes, we offer 24/7 customer support via chat and email."},
]

model = SentenceTransformer(MODEL_NAME)

questions = [f"passage: {entry['question']}" for entry in faq_entries] # passage prefix required by the model
question_embeddings = model.encode(
    questions,
    normalize_embeddings=True
)

# Build FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(np.array(question_embeddings))

# Save index + entries for retrieval
faiss.write_index(index, INDEX)
np.save(QUESTIONS, questions)