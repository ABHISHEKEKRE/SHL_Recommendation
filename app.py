from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import uvicorn
import faiss
import pickle
from keras_tuner.tuners import Hyperband
import tensorflow as tf
from sklearn.decomposition import PCA
app = FastAPI()
import os
from umap import UMAP
from flask import Flask, jsonify, request
import os
import json
import logging
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import dotenv
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
import uuid
import ast
import numpy as np


def run_rag(query):
    try:
        # Step 1: Extract metadata using Gemini
        extraction_result = extract_chain.invoke({"query": query})
        print(f"Raw extraction result: {extraction_result}")

        # Clean and parse the extracted metadata
        raw_text = extraction_result['text'].strip()
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            extracted_metadata = json.loads(json_match.group())
        else:
            print(f"Failed to find valid JSON in: {raw_text}")
            extracted_metadata = {
                "skills": [],
                "duration": 60,
                "languages": ["English (USA)"],
                "role_level": "",
                "test_type": [],
                "remote_testing": "no",
                "adaptive_testing": "no"
            }

        skills = extracted_metadata.get('skills', [])
        duration = int(extracted_metadata.get('duration', 60))
        languages = extracted_metadata.get('languages', ['English (USA)'])
        role_level = extracted_metadata.get('role_level', '')
        test_type = extracted_metadata.get('test_type', [])
        remote_testing = extracted_metadata.get('remote_testing', 'no')
        adaptive_testing = extracted_metadata.get('adaptive_testing', 'no')

        print(f"Extracted metadata - Skills: {skills}, Duration: {duration}, Languages: {languages}, Role Level: {role_level}, Test Type: {test_type}, Remote Testing: {remote_testing}, Adaptive Testing: {adaptive_testing}")

        # Step 2: Construct the query for Pinecone
        pinecone_query = f"assessment with skills {', '.join(skills)}"
        if role_level:
            pinecone_query += f" at {role_level} level"
        if test_type:
            pinecone_query += f" testing {', '.join(test_type)} skills"
        print(f"Pinecone query: {pinecone_query}")

        # Step 3: Retrieve exactly 10 assessments from Pinecone (unfiltered)
        results = vector_store.similarity_search(
            pinecone_query,
            k=10
        )
        print(f"Retrieved {len(results)} assessments")

        if len(results) < 10:
            print(f"Only {len(results)} assessments retrieved, expected 10. Check query or index data.")
        return results
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return f"Error: {str(e)}"

app = Flask(__name__)



@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get('query')

    # Validate request body
    # Initialize Pinecone
    try:
         pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
         print("Pinecone initialized successfully")
    except Exception as e:
         print(e)
         raise
    # Load the preprocessed DataFrame
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"CSV file not found at {CSV_PATH}")
        raise
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise
    # Initialize the embedding model for upsert
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"SentenceTransformer model {EMBEDDING_MODEL} initialized successfully")
    except Exception as e:
        print(f"Failed to initialize SentenceTransformer: {e}")
        raise
    # Initialize LangChain-compatible embeddings for PineconeVectorStore
    try:
        langchain_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"HuggingFaceEmbeddings for {EMBEDDING_MODEL} initialized successfully")
    except Exception as e:
        print(f"Failed to initialize HuggingFaceEmbeddings: {e}")
        raise
    try:
        index = pc.Index(INDEX_NAME)
        print(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        print(f"Failed to connect to index {INDEX_NAME}: {e}")
        raise
    try:
        vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=langchain_embeddings,
        text_key="text"
    )
        print("LangChain PineconeVectorStore initialized successfully")

        docs = vector_store.similarity_search("sample query", k=10)
        if docs:
             print(f"Sample retrieved document: {docs[0].metadata}")
        else:
             print("No documents retrieved from similarity search")
    except Exception as e:
             print(f"Failed to initialize PineconeVectorStore: {e}")
             raise
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
        print("Gemini model initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        raise
    # Prompt template for metadata extraction
    prompt_template_extract = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intelligent assistant. Extract the following information from the query:
- required skills
- assessment duration in minutes
- preferred languages (if specified)
- role level (e.g., entry, mid, senior, if specified)
- test type preferences (e.g., technical, behavioral, if specified)
- preference for remote testing (yes/no, if specified)
- preference for adaptive testing (yes/no, if specified)

Return ONLY the response in this JSON format, with no additional text or explanations:
{{
    "skills": ["...", "..."],
    "duration": ...,
    "languages": ["...", "..."],
    "role_level": "...",
    "test_type": ["...", "..."],
    "remote_testing": "...",
    "adaptive_testing": "..."
}}

Examples:
Query: We are looking for a candidate skilled in JavaScript and communication, entry level, for a 45-minute technical test in English, with remote testing.
Output: {{"skills": ["JavaScript", "communication"], "duration": 45, "languages": ["English"], "role_level": "entry", "test_type": ["technical"], "remote_testing": "yes", "adaptive_testing": "no"}}

Query: I need a candidate who knows Python, SQL, and teamwork, mid-level, for a 60-minute test in English and Spanish, with adaptive testing.
Output: {{"skills": ["Python", "SQL", "teamwork"], "duration": 60, "languages": ["English", "Spanish"], "role_level": "mid", "test_type": [], "remote_testing": "no", "adaptive_testing": "yes"}}

Now extract the data from this query:
Query: {query}
Output:
"""
)
      
# Build the extraction chain
    extract_chain = LLMChain(llm=llm, prompt=prompt_template_extract)
    extraction_result = extract_chain.invoke({"query": query})


# Configuration
os.environ["PINECONE_API_KEY"] = process.env.PINECODE_API_KEY
os.environ["GOOGLE_API_KEY"] = process.env.GOOGLE_API_KEY
INDEX_NAME = "shl-assessments"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Initialize embeddings
try:
    langchain_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
except Exception as e:
    raise

# Initialize Pinecone vector store
try:
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=langchain_embeddings,
        text_key="text"
    )
    print("LangChain PineconeVectorStore initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PineconeVectorStore: {e}")
    raise

# Initialize Gemini for extraction
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    print("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {e}")
    raise

# Prompt template for metadata extraction
prompt_template_extract = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intelligent assistant. Extract the following information from the query:
- required skills
- assessment duration in minutes
- preferred languages (if specified)
- role level (e.g., entry, mid, senior, if specified)
- test type preferences (e.g., technical, behavioral, if specified)
- preference for remote testing (yes/no, if specified)
- preference for adaptive testing (yes/no, if specified)

Return ONLY the response in this JSON format, with no additional text or explanations:
{{
    "skills": ["...", "..."],
    "duration": ...,
    "languages": ["...", "..."],
    "role_level": "...",
    "test_type": ["...", "..."],
    "remote_testing": "...",
    "adaptive_testing": "..."
}}

Examples:
Query: We are looking for a candidate skilled in JavaScript and communication, entry level, for a 45-minute technical test in English, with remote testing.
Output: {{"skills": ["JavaScript", "communication"], "duration": 45, "languages": ["English"], "role_level": "entry", "test_type": ["technical"], "remote_testing": "yes", "adaptive_testing": "no"}}

Query: I need a candidate who knows Python, SQL, and teamwork, mid-level, for a 60-minute test in English and Spanish, with adaptive testing.
Output: {{"skills": ["Python", "SQL", "teamwork"], "duration": 60, "languages": ["English", "Spanish"], "role_level": "mid", "test_type": [], "remote_testing": "no", "adaptive_testing": "yes"}}

Now extract the data from this query:
Query: {query}
Output:
"""
)

# Prompt template for response generation
prompt_template_generate = PromptTemplate(
    input_variables=["query", "context"],
    template="""
You are an intelligent assistant at SHL, specializing in recommending hiring assessments. Based on the user's query and the provided context about assessments, recommend exactly 10 suitable SHL assessments (or as many as available if fewer than 10 are retrieved). For each assessment, include:
- Assessment Name
- Assessment Type
- Description
- Role Level (if specified)
- Languages
- Test Type
- Length (in minutes)
- Remote Testing Support (yes/no)
- Adaptive Testing Support (yes/no)
- Fact Sheet Keywords

Ensure the response is detailed, clear, and tailored to the query. If fewer than 10 assessments are available, explain why and list all available ones. Do not include job role information in the recommendations.

Query: {query}

Context:
{context}

Response:
"""
)

# Build the extraction chain
extract_chain = LLMChain(llm=llm, prompt=prompt_template_extract)

# Example query
query = input("Enter your query: ")
response = run_rag(query)
print("Response:")
print(response)
#neww


# Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)