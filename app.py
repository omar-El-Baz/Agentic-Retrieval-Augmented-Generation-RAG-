# Import Streamlit for creating the web application UI.
import streamlit as st
# For operating system dependent functionality, like path checking and environment variables.
import os
# For working with JSON data, used to load pre-processed documents.
import json
# For generating unique IDs, though primarily used in the notebook to create processed_docs.json.
import uuid 
# For getting current date and time, used for timestamps in chat messages if needed.
from datetime import datetime
# For loading environment variables from a .env file.
from dotenv import load_dotenv

# --- Streamlit UI Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
# Sets page configurations like title, layout (wide), and initial sidebar state.
st.set_page_config(page_title="AI Travel Guide", layout="wide", initial_sidebar_state="expanded")

# --- Load Environment Variables (especially OPENAI_API_KEY) ---
# Load variables from .env file (e.g., OPENAI_API_KEY).
load_dotenv()
# Get the OpenAI API key from the environment.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Stop the app if the API key is not found, as the chatbot cannot function without it.
if not OPENAI_API_KEY:
    st.error("CRITICAL: OPENAI_API_KEY environment variable not set. The chatbot cannot function.")
    st.stop() # Halts execution of the Streamlit app.

# --- Backend Logic ---

# --- Member A: Data Ingestion & Vector DB Setup (Adapted for Streamlit) ---
# Use Streamlit's caching for resources that are expensive to create, like models or DB connections.
# These will be created once and reused across sessions/reruns unless inputs change.
@st.cache_resource 
def initialize_data_and_vector_db():
    # Imports are local to the function because they are only needed when this cached resource is first created.
    import pandas as pd 
    import nltk
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient, models as qdrant_models

    # Console log for tracking initialization steps (not shown in Streamlit UI directly).
    print("Attempting NLTK resource downloads (if not present)...")
    # Download NLTK 'punkt' for sentence tokenization if not found.
    try: nltk.data.find('tokenizers/punkt'); print("NLTK 'punkt' found.")
    except: nltk.download('punkt', quiet=True); print("NLTK 'punkt' downloaded.")
    # Download NLTK 'punkt_tab' if not found.
    try: nltk.data.find('tokenizers/punkt_tab'); print("NLTK 'punkt_tab' found.")
    except: nltk.download('punkt_tab', quiet=True); print("NLTK 'punkt_tab' downloaded.")

    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    # Load the sentence embedding model.
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model initialized.")

    print("Initializing Qdrant client (in-memory)...")
    # Initialize an in-memory Qdrant client for the vector database.
    qdrant_client_instance = QdrantClient(":memory:")
    # Define a name for the Qdrant collection.
    collection_name = "streamlit_travel_data_v2" 
    # Path to the pre-processed documents file.
    processed_docs_path = "processed_docs.json"
    print(f"Qdrant client initialized. Collection name: {collection_name}")

    documents_for_indexing = []
    # Check if the pre-processed documents file exists.
    if os.path.exists(processed_docs_path):
        try:
            # Load documents from the JSON file.
            with open(processed_docs_path, "r", encoding="utf-8") as f:
                documents_for_indexing = json.load(f)
            print(f"Successfully loaded {len(documents_for_indexing)} documents from {processed_docs_path}")
            # This file must contain documents where each 'id' field is a UUID string.
        except Exception as e:
            print(f"ERROR: Could not load pre-processed documents from {processed_docs_path}: {e}. RAG might not be effective.")
            documents_for_indexing = [] # Ensure list is empty on failure.
    else:
        print(f"WARNING: {processed_docs_path} not found. RAG will not have data. Run data processing (m1.ipynb) first.")
        documents_for_indexing = []

    # If documents were loaded, proceed to index them.
    if documents_for_indexing:
        print(f"Preparing to index {len(documents_for_indexing)} documents...")
        # Get the size of the embedding vectors from the model.
        vector_size = embedding_model.get_sentence_embedding_dimension()
        try:
            # If the collection already exists, delete it to ensure a fresh start.
            if qdrant_client_instance.collection_exists(collection_name=collection_name):
                 print(f"Collection '{collection_name}' exists. Deleting for recreation.")
                 qdrant_client_instance.delete_collection(collection_name=collection_name)
            
            # Create the new Qdrant collection with specified vector parameters.
            qdrant_client_instance.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE)
            )
            print(f"Qdrant collection '{collection_name}' created/recreated.")
            
            # Get content from each document.
            content_list = [doc.get("content", "") for doc in documents_for_indexing]
            # Filter for documents that have both content and an ID.
            valid_docs_for_embedding = [(doc, content) for doc, content in zip(documents_for_indexing, content_list) if content and doc.get("id")]
            
            if not valid_docs_for_embedding:
                print("WARNING: No valid documents with content and ID found for embedding after filtering.")
            else:
                print(f"Generating embeddings for {len(valid_docs_for_embedding)} valid documents...")
                # Prepare final lists of documents and content to be embedded.
                final_docs_to_index = [doc_tuple[0] for doc_tuple in valid_docs_for_embedding]
                final_content_list = [doc_tuple[1] for doc_tuple in valid_docs_for_embedding]

                # Generate embeddings for the document contents.
                embeddings = embedding_model.encode(final_content_list, show_progress_bar=False) # No progress bar in Streamlit.
                print("Embeddings generated.")
                
                # Prepare Qdrant points (vector + payload) for each document.
                points_to_upsert = [
                    qdrant_models.PointStruct(
                        id=doc["id"], # ID must be a UUID string.
                        vector=embeddings[i].tolist(),
                        payload={"text_content": doc.get("content",""), **doc.get("metadata",{})} # Store content and metadata.
                    )
                    for i, doc in enumerate(final_docs_to_index)
                ]

                if points_to_upsert:
                    # Add the points to the Qdrant collection.
                    qdrant_client_instance.upsert(collection_name=collection_name, points=points_to_upsert, wait=True)
                    print(f"Successfully indexed {len(points_to_upsert)} documents into Qdrant.")
                else:
                    print("No points were suitable for upserting into Qdrant after filtering.")
        except Exception as e:
            # Handle errors during Qdrant setup or indexing.
            print(f"ERROR: During Qdrant setup or indexing: {e}")
            return None, None, None # Return None to indicate failure.
    elif not documents_for_indexing and os.path.exists(processed_docs_path):
        print(f"INFO: {processed_docs_path} was loaded but found to be empty or contained no valid content/IDs for indexing.")

    # Return the initialized components.
    return embedding_model, qdrant_client_instance, collection_name

# --- Member B: CrewAI Agent and Crew Setup (Adapted for Streamlit) ---
# Cache the CrewAI components as well.
@st.cache_resource 
def initialize_crewai_components(_embedding_model, _qdrant_client, _qdrant_collection_name):
    # _embedding_model, etc., are passed to ensure this function runs after RAG setup if dependent.
    # Imports are local to the cached function.
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool # Corrected import path for BaseTool.
    from langchain_openai import ChatOpenAI

    print("Initializing CrewAI LLM...")
    # Initialize the Language Model for CrewAI.
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.2, openai_api_key=OPENAI_API_KEY)
    print("CrewAI LLM initialized.")

    # Define the custom tool for querying the vector database within the Streamlit app.
    class VectorDBQueryToolApp(BaseTool):
        name: str = "Travel Information Query Tool" # Tool name.
        description: str = "Queries a vector database for travel-related information like hotel reviews or itinerary details based on a user query. Input should be the user's query string." # Tool description for the agent.
        
        def _run(self, user_query: str) -> str:
            # Check if RAG components (embedding_model_st, qdrant_client_st) are ready.
            # These are accessed from the module level where they are initialized.
            if not all([embedding_model_st, qdrant_client_st]): 
                print("ERROR in VectorDBQueryToolApp: RAG components not ready.")
                return "Vector database is not available for search at the moment due to an initialization issue."
            
            print(f"[VectorDBQueryToolApp._run] Received query for DB search: '{user_query}'")
            # Use the Streamlit-specific search function.
            results = search_vector_db_st(user_query, top_k=3) 
            if not results: 
                return "No relevant information snippets were found in the database for this query."
            
            # Format the search results into a string for the agent.
            output_str = "Retrieved Information Snippets:\n"
            for i, res in enumerate(results):
                content_preview = res.get('content', 'N/A')[:150] + "..." if res.get('content') else 'N/A'
                output_str += f"Snippet {i+1} (Score: {res.get('score', 0.0):.2f}, Source: {res.get('metadata', {}).get('source','N/A')}): {content_preview}\n"
            return output_str

    print("Initializing CrewAI Tools and Agents...")
    # Create an instance of the custom tool.
    vector_db_tool_instance_st = VectorDBQueryToolApp()

    # Define the Retriever Agent.
    retriever_agent = Agent(
        role='Travel Information Retriever',
        goal='Use the "Travel Information Query Tool" to find data relevant to the user\'s query.',
        backstory='You are an AI assistant that efficiently queries a travel database.',
        tools=[vector_db_tool_instance_st], llm=llm, verbose=True, memory=False, allow_delegation=False # verbose=True for console logs
    )
    # Define the Summarizer Agent.
    summarizer_agent = Agent( 
        role='Summary Extractor',
        goal='Condense the retrieved information into a concise summary relevant to the user query.',
        backstory='You are an AI assistant skilled at summarizing text for clarity.',
        llm=llm, verbose=True, memory=False, allow_delegation=False
    )
    # Define the Composer Agent.
    composer_agent = Agent(
        role='Travel Advisor',
        goal='Formulate a helpful and friendly travel recommendation or answer based on the provided summary.',
        backstory='You are an AI travel assistant that provides engaging and practical advice.',
        llm=llm, verbose=True, memory=False, allow_delegation=False
    )
    print("CrewAI components initialized.")
    # Return the initialized agents and LLM.
    return retriever_agent, summarizer_agent, composer_agent, llm

# --- Search Function (uses globally initialized RAG components) ---
# This function performs the actual vector search.
def search_vector_db_st(query_text: str, top_k: int = 3) -> list[dict]:
    # Check if RAG components are ready.
    if not all([embedding_model_st, qdrant_client_st, qdrant_collection_name_st]):
        print("ERROR: Search components (embedding model or Qdrant client) not ready in Streamlit app.")
        return []
    # Validate the query.
    if not query_text or not isinstance(query_text, str) or not query_text.strip():
        print("ERROR: Invalid query text for vector DB search.")
        return []
    
    try:
        # Generate embedding for the query text.
        query_embedding = embedding_model_st.encode([query_text])[0]
    except Exception as e:
        print(f"ERROR: Encoding query in search_vector_db_st: {e}")
        return []

    try:
        # Search Qdrant. (Using .search for simplicity, .query or .query_points is newer).
        search_results_qdrant = qdrant_client_st.search( 
            collection_name=qdrant_collection_name_st,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        formatted_results = []
        # Format the Qdrant results.
        for hit in search_results_qdrant: 
            payload = hit.payload if hit.payload else {}
            formatted_results.append({
                "id": str(hit.id),
                "score": float(hit.score),
                "content": payload.get("text_content", ""), # Ensure this key matches the payload structure.
                "metadata": {k: v for k, v in payload.items() if k != "text_content"}
            })
        return formatted_results
    except Exception as e:
        # Handle errors during the Qdrant search.
        print(f"ERROR: During Qdrant search in Streamlit app: {e}")
        return []

# --- Initialize Components (Global scope for the app session) ---
# Define module-level variables to hold the initialized components.
embedding_model_st, qdrant_client_st, qdrant_collection_name_st = None, None, None
crew_retriever_st, crew_summarizer_st, crew_composer_st, crew_llm_st = None, None, None, None

# Initialize RAG components with a Streamlit spinner for UI feedback.
with st.spinner("Initializing data and vector database... This may take a moment..."):
    embedding_model_st, qdrant_client_st, qdrant_collection_name_st = initialize_data_and_vector_db()

# Initialize CrewAI components with a Streamlit spinner.
with st.spinner("Initializing AI Agents... This may take a moment..."):
    crew_retriever_st, crew_summarizer_st, crew_composer_st, crew_llm_st = initialize_crewai_components(
        embedding_model_st, qdrant_client_st, qdrant_collection_name_st
    )

# --- Crew Execution Function ---
# This function orchestrates the CrewAI agents to process a user query.
def get_travel_recommendation_streamlit(user_query: str) -> str:
    # Local import of CrewAI Task and Process, as they are only used here.
    from crewai import Task, Crew, Process 
    
    # Check if agents are ready.
    if not all([crew_retriever_st, crew_summarizer_st, crew_composer_st]):
        return "Error: CrewAI agents are not ready. Please check the application logs."
    # Validate the query.
    if not user_query or not user_query.strip(): 
        return "Please enter a query."

    # Define the task for the Retriever Agent.
    retrieval_task = Task(
        description=f"The user's query is: '{user_query}'. Use your tool to find relevant travel information from the database.",
        expected_output="A string containing relevant snippets from the travel database, or a message if no information is found.",
        agent=crew_retriever_st
    )
    # Define the task for the Summarizer Agent, which depends on the retrieval_task.
    summarization_task = Task(
        description=f"Based on the retrieved snippets for the query '{user_query}', create a concise summary. If no snippets were found, state that no information was available.",
        expected_output="A short, factual summary of the key information, or a statement that no information was found.",
        agent=crew_summarizer_st, context=[retrieval_task]
    )
    # Define the task for the Composer Agent, which depends on the summarization_task.
    composition_task = Task(
        description=f"Using the summary for the query '{user_query}', compose a friendly and helpful travel recommendation. If the summary indicates no information was found, politely inform the user.",
        expected_output="A final, well-formatted response for the user.",
        agent=crew_composer_st, context=[summarization_task]
    )
    # Create the Crew with the defined agents and tasks.
    travel_crew = Crew(
        agents=[crew_retriever_st, crew_summarizer_st, crew_composer_st],
        tasks=[retrieval_task, summarization_task, composition_task],
        process=Process.sequential, # Tasks are executed sequentially.
        verbose=True # Enable verbose logging for the crew in the console.
    )
    try:
        print("CrewAI kickoff initiated...")
        # Start the crew's process.
        result = travel_crew.kickoff()
        print("CrewAI kickoff finished.")
        return str(result) # Return the final result.
    except Exception as e:
        # Handle any errors during the crew's execution.
        print(f"ERROR: During CrewAI kickoff in Streamlit: {e}")
        return f"Sorry, I encountered an issue processing your request."

# --- Streamlit UI Rendering Starts Here ---
# Set the main title of the application.
st.title("🌍 AI Travel Guide Chatbot")
# Set a caption below the title.
st.caption("Your personal AI travel assistant. Ask me anything about your travel plans!")

# Initialize chat history in Streamlit's session state if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist with your travel plans today?"}]

# --- Sidebar UI Elements ---
with st.sidebar:
    st.header("Chat Controls")
    # Button to clear chat history.
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist with your travel plans today?"}]
        st.rerun() # Rerun the app to reflect the cleared history.
    st.markdown("---") # Horizontal line separator.
    st.header("About")
    st.markdown(
        "This chatbot uses a Retrieval-Augmented Generation (RAG) approach with CrewAI "
        "to provide travel recommendations. Ensure `processed_docs.json` (with UUIDs for IDs) "
        "is present in the app directory."
    )
    st.markdown("---")
    st.subheader("System Status")
    # Display status of key components.
    if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-") and len(OPENAI_API_KEY) > 40:
        st.success("OpenAI API Key: Loaded")
    else:
        st.error("OpenAI API Key: Invalid or missing")

    if embedding_model_st: st.success("Embedding Model: Initialized")
    else: st.warning("Embedding Model: Not initialized")
        
    if qdrant_client_st and qdrant_collection_name_st:
        try:
            # Check the status of the Qdrant collection.
            collection_info = qdrant_client_st.get_collection(collection_name=qdrant_collection_name_st)
            st.success(f"Vector DB: Connected. Indexed: {collection_info.points_count}")
        except Exception as e:
            # Handle cases where the collection might not be found or other errors.
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

# --- Main Chat Interface ---
# Display existing chat messages from session state.
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # Use "user" or "assistant" role for styling.
        st.markdown(message["content"]) # Display the message content.

# Input field for the user to type their query.
if prompt := st.chat_input("e.g., Find a luxury hotel in Paris near Eiffel Tower with a spa"):
    # Add user's message to chat history and display it.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the user's query and display the assistant's response.
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Create a placeholder for the assistant's response.
        
        # Show a spinner while the assistant is "thinking".
        with st.spinner("🤖 Assistant is thinking... This might take a few moments..."):
            response = "An unexpected issue occurred. Please try again." # Default response.
            # Check if all essential components are ready.
            if not all([crew_retriever_st, crew_summarizer_st, crew_composer_st, crew_llm_st]):
                 response = "I'm sorry, my AI components are not fully initialized. Please check the console logs or application status in the sidebar."
            # If RAG components are down but the query seems to need them, inform the user.
            elif not all([embedding_model_st, qdrant_client_st, qdrant_collection_name_st]) and ("hotel" in prompt.lower() or "review" in prompt.lower()): 
                 response = ("My document search capabilities (RAG) are not ready. "
                             "I can try to answer generally, but it won't be based on specific reviews. "
                             "Attempting general response...\n\n" + get_travel_recommendation_streamlit(prompt))
            else:
                # If all components are ready, get the recommendation.
                response = get_travel_recommendation_streamlit(prompt)
        
        # Display the actual response.
        message_placeholder.markdown(response)
        # Add assistant's response to chat history.
        st.session_state.messages.append({"role": "assistant", "content": response})