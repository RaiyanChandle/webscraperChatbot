import streamlit as st
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from rag_pipeline import scrape_website, extract_clean_text, chunk_text, store_in_qdrant, retrieve_context, generate_answer

st.set_page_config(page_title="RAG Knowledge Base", layout="wide")

# Load from .env if present
load_dotenv()

# Keys loaded directly from environment
qdrant_url = os.getenv("QDRANT_URL", "https://c84f661a-7be3-4501-ad5f-6fdc9675a501.eu-west-2-0.aws.cloud.qdrant.io:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
gemini_api_key = os.getenv("GEMINI_API_KEY", "")

with st.sidebar:
    st.title("Configuration")
    collection_name = st.text_input("Qdrant Collection Name", value="100xdevs_data")
    
    st.divider()
    
    st.subheader("Ingest Data")
    target_url = st.text_input("Website URL to Index", value="https://100xdevs.com/")
    if st.button("Scrape & Index"):
        if not qdrant_url or not qdrant_api_key:
            st.error("Missing Qdrant Credentials.")
        else:
            with st.spinner("Processing... Document extraction and embedding..."):
                try:
                    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                    html = scrape_website(target_url)
                    if html:
                        clean_text = extract_clean_text(html)
                        st.info("Website content successfully extracted.")
                        
                        chunks = chunk_text(clean_text)
                        st.info(f"Generated {len(chunks)} text chunks.")
                        
                        store_in_qdrant(client, collection_name, chunks, target_url)
                        st.success("Successfully vectorized and stored in Qdrant!")
                    else:
                        st.error("Failed to fetch the valid HTML from URL.")
                except Exception as e:
                    st.error(f"Encountered an exception: {e}")

st.title("Chat with RAG AI")
st.write("A knowledge assistant that answers strictly on context.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Show message history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the website (e.g. 'What services do they offer?'):"):
    # Render user query
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing
    with st.chat_message("assistant"):
        if not qdrant_url or not qdrant_api_key or not gemini_api_key:
            st.warning("Please configure your API keys (Qdrant & Gemini) in the sidebar.")
        else:
            with st.spinner("Searching Vector Database..."):
                try:
                    q_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                    # Get contexts
                    retrieved_chunks = retrieve_context(q_client, collection_name, prompt)
                    
                    if retrieved_chunks:
                        with st.spinner("Generating LLM Response..."):
                            answer = generate_answer(prompt, retrieved_chunks, gemini_api_key)
                        st.markdown(answer)
                        
                        # Add citations for verify
                        with st.expander("Show Retrieved Chunks"):
                            for idx, c in enumerate(retrieved_chunks):
                                st.write(f"**Chunk {idx+1}:** {c}")
                                
                        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    else:
                        st.warning("Found no matching context for your query.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
