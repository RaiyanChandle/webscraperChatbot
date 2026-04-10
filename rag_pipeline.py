import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import google.generativeai as genai
import uuid

# Load local embedding model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
VECTOR_SIZE = 384  # Default output size for all-MiniLM-L6-v2

def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def extract_clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def chunk_text(text, target_words=300):
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= target_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def generate_embeddings(chunks):
    return model.encode(chunks)

def store_in_qdrant(qdrant_client, collection_name, chunks, url):
    # Ensure collection exists, if not create it
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    
    embeddings = generate_embeddings(chunks)
    
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={"text": chunk, "source": url}
            )
        )
    
    # Upsert the new vectors
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    return True

def retrieve_context(qdrant_client, collection_name, query, top_k=3):
    query_vector = model.encode(query).tolist()
    
    # Newer versions of QdrantClient use query_points
    try:
        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        )
        return [hit.payload["text"] for hit in search_result.points]
    except AttributeError:
        # Fallback for older versions
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return [hit.payload["text"] for hit in search_result]

def generate_answer(query, contexts, gemini_api_key):
    genai.configure(api_key=gemini_api_key)
    # Using Gemini 2.5 Flash as the latest stable model for the newest SDK/API
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    
    context_str = "\n\n---\n\n".join(contexts)
    
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. 

Strict Rules:
- You must ONLY use the provided context to answer the question.
- Do not use outside knowledge or hallucinate information.
- If the answer to the question is not present in the context, confidently state: "I do not have enough information to answer that based on the provided context."

Context Information:
{context_str}

User Question:
{query}

Answer:"""

    response = gemini_model.generate_content(prompt)
    return response.text
