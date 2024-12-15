import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import spacy
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy NER model
@st.cache_resource
def load_medical_ner():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("Medical NER model not found. Please install 'en_core_web_sm'.")
        return None

# Entity extraction function
def extract_medical_entities(text, nlp):
    doc = nlp(text)
    
    entities = {
        'Symptoms': [],
        'Diseases': [],
        'Treatments': []
    }
    
    # Extract named entities and custom keywords
    medical_keywords = {
        'Symptoms': ['pain', 'fever', 'cough', 'headache', 'fatigue', 'inflammation'],
        'Diseases': ['diabetes', 'cancer', 'hypertension', 'flu', 'covid', 'pneumonia'],
        'Treatments': ['medication', 'surgery', 'therapy', 'vaccine', 'antibiotics']
    }
    
    # Named entity extraction
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
            continue
        entities['Diseases'].append(ent.text)
    
    # Keyword matching
    text_lower = text.lower()
    for category, keywords in medical_keywords.items():
        matches = [kw for kw in keywords if kw in text_lower]
        entities[category].extend(matches)
    
    return entities

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="🩺")

# Title
st.title("🩺 Medical Q&A Chatbot with Entity Recognition")

# Get API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize session state for vector store
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Function to create embedding and vector store
def create_vector_store():
    csv_path = "processed_data/medical_qa.csv"
    
    try:
        # Read CSV and take last 5000 rows
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        df_subset = df.tail(5000)
        
        if df_subset.empty:
            raise ValueError("No data found in the subset")
        
        documents = []
        for _, row in df_subset.iterrows():
            content = f"Question: {row.get('question', '')} \nAnswer: {row.get('answer', '')}"
            metadata = {
                'focus': row.get('focus', 'Unknown'),
                'source': row.get('source', 'Unknown'),
                'question': row.get('question', ''),
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        
        huggingface_embeddings = HuggingFaceBgeEmbeddings(
            model_name="NeuML/pubmedbert-base-embeddings",
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = FAISS.from_documents(
            documents=split_docs, 
            embedding=huggingface_embeddings
        )
        
        st.write(f"Total original documents: {len(df)}")
        st.write(f"Documents in subset: {len(documents)}")
        st.write(f"Total chunks created: {len(split_docs)}")
        
        return vectorstore
    
    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        st.error(f"Error creating vector store: {e}")
        st.error(f"Ensure the CSV file exists at {csv_path}")
        
        return None

# Embedding button
if st.button("Create Vector Store (Last 5000 Rows)"):
    with st.spinner("Creating Vector Store..."):
        st.session_state.vectorstore = create_vector_store()
        if st.session_state.vectorstore:
            st.success("Vector Store Created Successfully!")

# Prompt Template
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful medical assistant trained to provide accurate and helpful medical information.
Use only the context provided to answer the question. If the answer is not in the context, 
politely explain that you cannot find the specific information.

Context:
{context}

Question: {input}

Answer:
""")

# Chat Interface
def medical_chatbot():
    # Load NER model
    nlp = load_medical_ner()
    
    # Check if API keys are available
    if not GROQ_API_KEY:
        st.error("Groq API Key is missing. Please check your .env file.")
        return

    # Initialize LLM 
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="mixtral-8x7b-32768"
    )

    # Check if vector store is created
    if st.session_state.vectorstore is None:
        st.warning("Please create the Vector Store first.")
        return

    # User input
    query = st.text_input("Enter your medical question:")
    
    if query and nlp:
        try:
            # Extract medical entities
            entities = extract_medical_entities(query, nlp)
            
            # Display extracted entities
            st.subheader("Extracted Entities")
            for category, values in entities.items():
                if values:
                    st.markdown(f"**{category}:** {', '.join(set(values))}")
            
            # Create document chain
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            
            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(
                st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}), 
                document_chain
            )
            
            # Process query
            with st.spinner("Searching for the most relevant information..."):
                start_time = time.time()
                response = retrieval_chain.invoke({"input": query})
                processing_time = time.time() - start_time
            
            # Display response
            st.subheader("Answer:")
            st.write(response['answer'])
            
            # Show source documents
            with st.expander("Source Documents"):
                for doc in response['context']:
                    st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    st.write(doc.page_content)
                    st.write("---")
            
            # Show processing time
            st.caption(f"Processing Time: {processing_time:.2f} seconds")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Run the chatbot
medical_chatbot()

# Footer
st.markdown("---")
st.markdown("*Disclaimer: This chatbot provides informational medical guidance. Always consult a healthcare professional for medical advice.*")