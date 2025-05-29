# --- START OF FILE app.py ---

import streamlit as st
import os
import json
import uuid # For potential future use or if regenerating IDs in-app
from datetime import datetime
from dotenv import load_dotenv

# --- Streamlit UI Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="AI Travel Guide", layout="wide", initial_sidebar_state="expanded")

# --- Load Environment Variables (especially OPENAI_API_KEY) ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("CRITICAL: OPENAI_API_KEY environment variable not set (or .env file not found/configured). The chatbot cannot function.")
    st.stop()

# --- Backend Logic ---

# --- Member A: Data Ingestion & Vector DB Setup (Adapted for Streamlit) ---
@st.cache_resource # Cache heavy resources
def initialize_data_and_vector_db():
    import pandas as pd # Local import for cached function
    import nltk
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient, models as qdrant_models

    print("Attempting NLTK resource downloads (if not present)...")
    try: nltk.data.find('tokenizers/punkt'); print("NLTK 'punkt' found.")
    except: nltk.download('punkt', quiet=True); print("NLTK 'punkt' downloaded.")
    
    try: nltk.data.find('tokenizers/punkt_tab'); print("NLTK 'punkt_tab' found.")
    except: nltk.download('punkt_tab', quiet=True); print("NLTK 'punkt_tab' downloaded.")

    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model initialized.")

    print("Initializing Qdrant client (in-memory)...")
    qdrant_client_instance = QdrantClient(":memory:")
    collection_name = "streamlit_travel_data_v2" # Use a distinct name
    processed_docs_path = "processed_docs.json"
    print(f"Qdrant client initialized. Collection name: {collection_name}")

    documents_for_indexing = []
    if os.path.exists(processed_docs_path):
        try:
            with open(processed_docs_path, "r", encoding="utf-8") as f:
                documents_for_indexing = json.load(f)
            print(f"Successfully loaded {len(documents_for_indexing)} documents from {processed_docs_path}")
            # CRITICAL: Ensure this file's 'id' fields are UUIDs.
        except Exception as e:
            print(f"ERROR: Could not load pre-processed documents from {processed_docs_path}: {e}. RAG might not be effective.")
            documents_for_indexing = []
    else:
        print(f"WARNING: {processed_docs_path} not found. RAG will not have data. Run data processing (m1.ipynb) first.")
        documents_for_indexing = []

    if documents_for_indexing:
        print(f"Preparing to index {len(documents_for_indexing)} documents...")
        vector_size = embedding_model.get_sentence_embedding_dimension()
        try:
            if qdrant_client_instance.collection_exists(collection_name=collection_name):
                 print(f"Collection '{collection_name}' exists. Deleting for recreation.")
                 qdrant_client_instance.delete_collection(collection_name=collection_name)
            
            qdrant_client_instance.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE)
            )
            print(f"Qdrant collection '{collection_name}' created/recreated.")
            
            content_list = [doc.get("content", "") for doc in documents_for_indexing]
            valid_docs_for_embedding = [(doc, content) for doc, content in zip(documents_for_indexing, content_list) if content and doc.get("id")]
            
            if not valid_docs_for_embedding:
                print("WARNING: No valid documents with content and ID found for embedding after filtering.")
            else:
                print(f"Generating embeddings for {len(valid_docs_for_embedding)} valid documents...")
                final_docs_to_index = [doc_tuple[0] for doc_tuple in valid_docs_for_embedding]
                final_content_list = [doc_tuple[1] for doc_tuple in valid_docs_for_embedding]

                embeddings = embedding_model.encode(final_content_list, show_progress_bar=False)
                print("Embeddings generated.")
                
                points_to_upsert = [
                    qdrant_models.PointStruct(
                        id=doc["id"], # MUST be a UUID string from processed_docs.json
                        vector=embeddings[i].tolist(),
                        payload={"text_content": doc.get("content",""), **doc.get("metadata",{})}
                    )
                    for i, doc in enumerate(final_docs_to_index)
                ]

                if points_to_upsert:
                    qdrant_client_instance.upsert(collection_name=collection_name, points=points_to_upsert, wait=True)
                    print(f"Successfully indexed {len(points_to_upsert)} documents into Qdrant.")
                else:
                    print("No points were suitable for upserting into Qdrant after filtering.")
        except Exception as e:
            print(f"ERROR: During Qdrant setup or indexing: {e}")
            return None, None, None 
    elif not documents_for_indexing and os.path.exists(processed_docs_path):
        print(f"INFO: {processed_docs_path} was loaded but found to be empty or contained no valid content/IDs for indexing.")

    return embedding_model, qdrant_client_instance, collection_name

# --- Member B: CrewAI Agent and Crew Setup (Adapted for Streamlit) ---
@st.cache_resource 
def initialize_crewai_components(_embedding_model, _qdrant_client, _qdrant_collection_name):
    # Pass RAG components if tool needs them, ensures they are initialized first
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool # Corrected import
    from langchain_openai import ChatOpenAI

    print("Initializing CrewAI LLM...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.2, openai_api_key=OPENAI_API_KEY)
    print("CrewAI LLM initialized.")

    # Define the tool class inside the function to use the passed RAG components
    class VectorDBQueryToolApp(BaseTool):
        name: str = "Travel Information Query Tool"
        description: str = "Queries a vector database for travel-related information like hotel reviews or itinerary details based on a user query. Input should be the user's query string."
        
        # Store RAG components on the instance if needed, or access them from the outer scope (if careful with Streamlit's execution model)
        # For simplicity here, search_vector_db_st will use the globally available (module-level) RAG components.
        
        def _run(self, user_query: str) -> str:
            # Access module-level RAG components directly
            # This relies on them being initialized before this tool is used.
            if not all([embedding_model_st, qdrant_client_st]): 
                print("ERROR in VectorDBQueryToolApp: RAG components not ready.")
                return "Vector database is not available for search at the moment due to an initialization issue."
            
            print(f"[VectorDBQueryToolApp._run] Received query for DB search: '{user_query}'")
            results = search_vector_db_st(user_query, top_k=3) # search_vector_db_st uses global RAG components
            if not results: 
                return "No relevant information snippets were found in the database for this query."
            
            output_str = "Retrieved Information Snippets:\n"
            for i, res in enumerate(results):
                content_preview = res.get('content', 'N/A')[:150] + "..." if res.get('content') else 'N/A'
                output_str += f"Snippet {i+1} (Score: {res.get('score', 0.0):.2f}, Source: {res.get('metadata', {}).get('source','N/A')}): {content_preview}\n"
            return output_str

    print("Initializing CrewAI Tools and Agents...")
    vector_db_tool_instance_st = VectorDBQueryToolApp()

    retriever_agent = Agent(
        role='Travel Information Retriever',
        goal='Use the "Travel Information Query Tool" to find data relevant to the user\'s query.',
        backstory='You are an AI assistant that efficiently queries a travel database.',
        tools=[vector_db_tool_instance_st], llm=llm, verbose=True, memory=False, allow_delegation=False
    )
    summarizer_agent = Agent(
        role='Summary Extractor',
        goal='Condense the retrieved information into a concise summary relevant to the user query.',
        backstory='You are an AI assistant skilled at summarizing text for clarity.',
        llm=llm, verbose=True, memory=False, allow_delegation=False
    )
    composer_agent = Agent(
        role='Travel Advisor',
        goal='Formulate a helpful and friendly travel recommendation or answer based on the provided summary.',
        backstory='You are an AI travel assistant that provides engaging and practical advice.',
        llm=llm, verbose=True, memory=False, allow_delegation=False
    )
    print("CrewAI components initialized.")
    return retriever_agent, summarizer_agent, composer_agent, llm

# --- Search Function (uses globally initialized RAG components) ---
def search_vector_db_st(query_text: str, top_k: int = 3) -> list[dict]:
    if not all([embedding_model_st, qdrant_client_st, qdrant_collection_name_st]):
        print("ERROR: Search components (embedding model or Qdrant client) not ready in Streamlit app.")
        return []
    if not query_text or not isinstance(query_text, str) or not query_text.strip():
        print("ERROR: Invalid query text for vector DB search.")
        return []
    
    try:
        query_embedding = embedding_model_st.encode([query_text])[0]
    except Exception as e:
        print(f"ERROR: Encoding query in search_vector_db_st: {e}")
        return []

    try:
        search_results_qdrant = qdrant_client_st.search( # or .query if using newer alias
            collection_name=qdrant_collection_name_st,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        formatted_results = []
        # Qdrant search returns a list of ScoredPoint objects
        for hit in search_results_qdrant: 
            payload = hit.payload if hit.payload else {}
            formatted_results.append({
                "id": str(hit.id),
                "score": float(hit.score),
                "content": payload.get("text_content", ""), # Standardized to "text_content"
                "metadata": {k: v for k, v in payload.items() if k != "text_content"}
            })
        return formatted_results
    except Exception as e:
        print(f"ERROR: During Qdrant search in Streamlit app: {e}")
        # st.warning(f"RAG search failed: {e}") # This would cause set_page_config error if called here
        return []

# --- Initialize Components (with UI feedback via spinner) ---
# These are module-level variables that will hold the initialized components
embedding_model_st, qdrant_client_st, qdrant_collection_name_st = None, None, None
crew_retriever_st, crew_summarizer_st, crew_composer_st, crew_llm_st = None, None, None, None

with st.spinner("Initializing data and vector database... This may take a moment..."):
    embedding_model_st, qdrant_client_st, qdrant_collection_name_st = initialize_data_and_vector_db()

with st.spinner("Initializing AI Agents... This may take a moment..."):
    # Pass the RAG components to ensure they are available if the tool needs them during its own init
    # However, our current tool's _run method accesses them globally.
    crew_retriever_st, crew_summarizer_st, crew_composer_st, crew_llm_st = initialize_crewai_components(
        embedding_model_st, qdrant_client_st, qdrant_collection_name_st
    )

# --- Crew Execution Function ---
def get_travel_recommendation_streamlit(user_query: str) -> str:
    from crewai import Task, Crew, Process 
    
    if not all([crew_retriever_st, crew_summarizer_st, crew_composer_st]):
        return "Error: CrewAI agents are not ready. Please check the application logs."
    if not user_query or not user_query.strip(): 
        return "Please enter a query."

    retrieval_task = Task(
        description=f"The user's query is: '{user_query}'. Use your tool to find relevant travel information from the database.",
        expected_output="A string containing relevant snippets from the travel database, or a message if no information is found.",
        agent=crew_retriever_st
    )
    summarization_task = Task(
        description=f"Based on the retrieved snippets for the query '{user_query}', create a concise summary. If no snippets were found, state that no information was available.",
        expected_output="A short, factual summary of the key information, or a statement that no information was found.",
        agent=crew_summarizer_st, context=[retrieval_task]
    )
    composition_task = Task(
        description=f"Using the summary for the query '{user_query}', compose a friendly and helpful travel recommendation. If the summary indicates no information was found, politely inform the user.",
        expected_output="A final, well-formatted response for the user.",
        agent=crew_composer_st, context=[summarization_task]
    )
    travel_crew = Crew(
        agents=[crew_retriever_st, crew_summarizer_st, crew_composer_st],
        tasks=[retrieval_task, summarization_task, composition_task],
        process=Process.sequential, 
        verbose=True # Boolean: True for detailed logs, False for silent. Can also be integer 0, 1, 2.
    )
    try:
        # st.write("🤖 CrewAI is processing your request...") # This would cause error if called before main UI rendering
        print("CrewAI kickoff initiated...")
        result = travel_crew.kickoff()
        print("CrewAI kickoff finished.")
        return str(result)
    except Exception as e:
        print(f"ERROR: During CrewAI kickoff in Streamlit: {e}")
        # st.error(f"Sorry, I encountered an issue processing your request. Details: {str(e)}")
        return f"Sorry, I encountered an issue processing your request."

# --- Streamlit UI Rendering Starts Here ---
st.title("🌍 AI Travel Guide Chatbot")
st.caption("Your personal AI travel assistant. Ask me anything about your travel plans!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist with your travel plans today?"}]

# Sidebar for Info & Controls
with st.sidebar:
    st.header("Chat Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist with your travel plans today?"}]
        st.rerun()
    st.markdown("---")
    st.header("About")
    st.markdown(
        "This chatbot uses a Retrieval-Augmented Generation (RAG) approach with CrewAI "
        "to provide travel recommendations. Ensure `processed_docs.json` (with UUIDs for IDs) "
        "is present in the app directory."
    )
    st.markdown("---")
    st.subheader("System Status")
    if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-") and len(OPENAI_API_KEY) > 40:
        st.success("OpenAI API Key: Loaded")
    else:
        st.error("OpenAI API Key: Invalid or missing")

    if embedding_model_st: st.success("Embedding Model: Initialized")
    else: st.warning("Embedding Model: Not initialized")
        
    if qdrant_client_st and qdrant_collection_name_st:
        try:
            collection_info = qdrant_client_st.get_collection(collection_name=qdrant_collection_name_st)
            st.success(f"Vector DB: Connected. Indexed: {collection_info.points_count}")
        except Exception as e:
            if "not found" in str(e).lower() or "status_code=404" in str(e).lower() or "not exist" in str(e).lower() :
                 st.warning(f"Vector DB: Collection '{qdrant_collection_name_st}' not found or empty.")
            else:
                st.warning(f"Vector DB: Status unknown ({e})")
    else:
        st.warning("Vector DB: Not initialized")

    if crew_retriever_st and crew_summarizer_st and crew_composer_st and crew_llm_st:
        st.success("CrewAI Agents: Initialized")
    else:
        st.warning("CrewAI Agents: Not fully initialized")
    
    st.markdown("---")
    st.markdown("Powered by [Streamlit](https://streamlit.io/) & [CrewAI](https://www.crewai.com/)")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("e.g., Find a luxury hotel in Paris near Eiffel Tower with a spa"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Provide initial feedback
        # message_placeholder.markdown("Thinking... ⏳") # Alternative to spinner
        
        with st.spinner("🤖 Assistant is thinking... This might take a few moments..."):
            response = "An unexpected issue occurred. Please try again." # Default response
            if not all([crew_retriever_st, crew_summarizer_st, crew_composer_st, crew_llm_st]):
                 response = "I'm sorry, my AI components are not fully initialized. Please check the console logs or application status in the sidebar."
            elif not all([embedding_model_st, qdrant_client_st, qdrant_collection_name_st]) and ("hotel" in prompt.lower() or "review" in prompt.lower()): # Basic check if RAG might be needed
                 response = ("My document search capabilities (RAG) are not ready. "
                             "I can try to answer generally, but it won't be based on specific reviews. "
                             "Attempting general response...\n\n" + get_travel_recommendation_streamlit(prompt))
            else:
                response = get_travel_recommendation_streamlit(prompt)
        
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})