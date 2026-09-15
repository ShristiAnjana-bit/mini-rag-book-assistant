from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#1. Books read
model = SentenceTransformer("all-MiniLM-L6-v2")
with open("data/books.txt" , "r") as file:
    text = file.read()

#2.chunks
chunks = text.split("\n\n")


#3.Book embeddings
embeddings = []
for chunk in chunks:
    embedding = model.encode(chunk)
    embeddings.append(embedding)

#4.User query
query = "Which book has a detective and mystery story?"

#5.Query embedding
query_embedding = model.encode(query)

print(query_embedding.shape)

scores = []

for i, embedding in enumerate(embeddings):
 
    score = cosine_similarity([query_embedding],[embedding])
    scores.append(score)
    
highest = max(scores)
highest_index = scores.index(highest)

print(highest_index)
retrieved_chunk = chunks[highest_index]
print(retrieved_chunk)
    
