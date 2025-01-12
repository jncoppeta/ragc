import ast
import json
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, Index, MilvusClient, connections, utility
import numpy as np
import dotenv
import pdfplumber
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import logging
import re
import nltk
nltk.download('punkt_tab')

# Configure logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO,  # You can change this to DEBUG for more detailed logs
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Load environment variables
dotenv.load_dotenv()

# Ollama embed function
def embed(text: str):
    url = "http://ollama:11434/api/embed"  # Pointing to Ollama container
    headers = {"Content-Type": "application/json"}

    data = {
        "model": "all-minilm",
        "input": text
    }

    logger.debug(f"Sending embedding request for text: {text[:50]}...")  # Log first 50 characters for debugging
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        try:
            embeddings = response.json()['embeddings'][0]
            logger.debug(f"Embedding received successfully for text: {text[:50]}")
            return embeddings
        except Exception as e:
            logger.error(f"Error parsing embedding response: {e}")
            return f"Error ({response.status_code}): {response.text}"
    else:
        logger.error(f"Error from Ollama embedding service ({response.status_code}): {response.text}")
        return f"Error ({response.status_code}): {response.text}"

# Connect to Milvus
connections.connect(alias="default", host=os.getenv('MILVUS_HOST', 'milvus-standalone'), port=os.getenv('MILVUS_PORT', '19530'))

# Define the collection name
collection_name = os.getenv('COLLECTION_NAME', 'my_collection')  # Default collection name

# Check if collection exists, if not create it
try:
    # Check if the collection exists
    if not utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' does not exist. Creating it now.")

        # Define the schema for the collection
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384, description="vector"),  # Specify dim in params
                FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=2000)
            ]
        )
        
        # Create the collection
        milvus_client = Collection(name=collection_name, schema=schema)
        logger.info(f"Collection '{collection_name}' created.")
        logger.info(f"Schema: {milvus_client.schema}")
        
        # Create an index for the embedding field
        index_params = {
            "index_type": "IVF_FLAT",  # Choose appropriate index type
            "metric_type": "L2",       # Metric for similarity
            "params": {"nlist": 128}   # Tuning parameter
        }
        index = Index(collection=milvus_client, field_name="embedding", index_params=index_params)
        logger.info("Index created successfully.")
        
        # Load the collection into memory
        milvus_client.load()
        logger.info("Collection loaded into memory.")
    else:
        # Connect to the existing collection
        milvus_client = Collection(collection_name)
        logger.info(f"Collection '{collection_name}' already exists.")
        logger.info(f"Schema: {milvus_client.schema}")
        
        # Ensure the collection is loaded into memory
        milvus_client.load()
        logger.info("Collection loaded into memory.")

except Exception as e:
    logger.error(f"Error while handling collection: {e}")


# Function to extract text from PDF
def get_text(pdf_path):
    logger.info(f"Extracting text from PDF: {pdf_path}...")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    sentences = sent_tokenize(text)
    logger.info(f"Extracted {len(sentences)} sentences from PDF.")
    return [sent.strip() for sent in sentences if len(sent.strip()) > 5]

# Function to chunk the document
def chunk_document(text, max_length=2000):
    sentences = re.split(r'(?<=\.)\s', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding the sentence exceeds the max length, start a new chunk
        if len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk)
                logger.debug(f"Chunk added: {current_chunk[:50]}...")  # Log the first 50 characters
            current_chunk = sentence
        else:
            # Add the sentence to the current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if any
    if current_chunk:
        chunks.append(current_chunk)
        logger.debug(f"Final chunk added: {current_chunk[:50]}...")  # Log the first 50 characters
    
    return chunks

# Function to add data to the collection (Embedding the data and inserting into Milvus)
def add_data_to_collection(all_text):
    try:
        embeddings = []
        logger.info("Starting embedding process...")

        # Generate embeddings for each document
        for text in all_text:
            logger.debug(f"Generating embedding for text: {text[:50]}...")  # Log first 50 characters of text for debugging
            vector = embed(text)  # Assuming this returns a list of floats (embedding vector)
            if vector and isinstance(vector, list):  # Ensure that the embedding is valid and a list of floats
                embeddings.append(vector)

        # Ensure embeddings and documents match in length
        if len(embeddings) != len(all_text):
            logger.error(f"Embeddings count ({len(embeddings)}) does not match text count ({len(all_text)}).")
            raise ValueError(f"Embeddings count ({len(embeddings)}) does not match text count ({len(all_text)}).")

        logger.info(f"Successfully generated embeddings for {len(all_text)} documents.")

        # Prepare the entities for insertion into Milvus
        entities = [
            embeddings,  # This should correspond to 'embedding'
            all_text  # This should correspond to 'document'
        ]
        
        # Insert data into Milvus collection
        logger.info(f"Inserting {len(all_text)} documents into the collection.")
        milvus_client.insert([entities[0], entities[1]])  # Insert embeddings and documents as separate lists
        logger.info(f"Successfully inserted {len(all_text)} documents into the collection.")
        
    except Exception as e:
        logger.error(f"Error inserting data into collection: {e}")
        raise  # Re-raise the exception after logging it

# Add PDF data to the collection
pdf_folder = "./pdfs"  # Specify the folder containing PDFs
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

all_text = []
for pdf in pdf_files:
    logger.info(f"Processing {pdf}...")
    pdf_text = get_text(pdf)
    all_text.extend(pdf_text)  # Add extracted sentences to the data list

# Now chunk the text into smaller pieces
logger.info("Chunking the text into smaller documents...")
chunked_text = []
for text in all_text:
    chunked_text.extend(chunk_document(text))

# Add the chunked data to Milvus
add_data_to_collection(chunked_text)

# Define input schema for the /question route
class QuestionRequest(BaseModel):
    question: str

# Function to query Ollama model with the context and question
def generate(query: str):
    url = "http://ollama:11434/api/generate"  # Pointing to Ollama container
    headers = {"Content-Type": "application/json"}

    data = {
        "model": "mistral:7b",
        "prompt": query,
        "max_tokens": 50,
        "stream": False
    }

    logger.debug(f"Sending generate request for query: {query[:50]}...")  # Log first 50 characters for debugging
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        try:
            response_text = json.loads(response.text)['response']
            logger.debug(f"Received response from Ollama model: {response_text[:50]}...")  # Log first 50 characters
            return response_text
        except Exception as e:
            logger.error(f"Error parsing generate response: {e}")
            return f"Error (): {e}"
    else:
        logger.error(f"Error from Ollama generate service ({response.status_code}): {response.text}")
        return f"Error ({response.status_code}): {response.text}"

@app.get("/health")
async def health_check():
    return {
        "status_code": 200,
        "message": "healthy"
    }

@app.post("/question")
async def question(request: QuestionRequest):
    # Step 1: Generate embedding for the question
    logger.info(f"Generating embedding for question: {request.question[:50]}...")  # Log first 50 characters for debugging
    query_embedding = embed(request.question)

    # Step 2: Search for the 5 most similar vectors in Milvus
    logger.info(f"Searching for similar vectors in Milvus for question: {request.question[:50]}...")
    
    # Specify the field name for embeddings, which should match your collection schema
    anns_field = "embedding"  # This is the name of the field in your collection that stores embeddings
    
    # Define search parameters (use Euclidean distance or other metric depending on your data)
    param = {
        "metric_type": "L2",  # L2 distance (Euclidean distance)
        "params": {"nprobe": 10}  # Adjust the number of probes for optimization
    }
    
    # Perform the search
    search_results = milvus_client.search(
        data=[query_embedding],  # Convert the question to an embedding vector
        anns_field=anns_field,  # This is the field name in your collection for embeddings
        param=param,  # Search parameters with inner product metric
        limit=7,  # Return top 7 results
        output_fields=["document"],  # Return the text field
    )
    

    client = MilvusClient(
        uri="http://milvus-standalone:19530",
        token="root:Milvus"
    )

    texts = set()
    for id in search_results[0].ids:
        res = client.get(
            collection_name=os.getenv("COLLECTION_NAME"),
            ids=[id],
            output_fields=["document"]
        )
        logger.info(res)
        logger.info(res[0])
        texts.add(res[0]['document'])


    # Convert top responses into a string
    context = "\n".join(
        [text for text in texts]
    )
    logger.info(context)

    # Step 4: Query Ollama model with the question and context
    logger.info(f"Querying Ollama model with context and question...")
    response_text = generate(f"{context}\n{request.question}")

    return {"response": response_text}

