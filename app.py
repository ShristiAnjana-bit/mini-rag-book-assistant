from dotenv import load_dotenv
import os

from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# 1. Books read
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/books.txt", "r") as file:
    text = file.read()


# 2. Chunks
chunks = text.split("\n\n")


# 3. Book embeddings
embeddings = []

for chunk in chunks:
    embedding = model.encode(chunk)
    embeddings.append(embedding)


# 4. User query
query = "Which book has a detective and mystery story?"


# 5. Query embedding
query_embedding = model.encode(query)

print(query_embedding.shape)


# 6. Similarity scores
scores = []

for i, embedding in enumerate(embeddings):
    score = cosine_similarity([query_embedding], [embedding])
    scores.append(score)


# 7. Find highest similarity
highest = max(scores)
highest_index = scores.index(highest)

print(highest_index)


# 8. Retrieve the most relevant chunk
retrieved_chunk = chunks[highest_index]

print(retrieved_chunk)


# 9. Create RAG prompt
prompt = f"Question: {query}\nContext: {retrieved_chunk}"

print(prompt)


# 10. Send context + question to Gemini
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print(response.text)