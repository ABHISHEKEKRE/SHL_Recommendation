from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import re
import logging
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pinecone import Pinecone

# Initialize FastAPI app and logging
app = FastAPI(title="SHL Assessment Recommendation API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
INDEX_NAME = "shl-assessments"
EMBEDDING_MODEL = "models/all-MiniLM-L6-v2"
ASSESSMENT_LINK_CSV_PATH = "finalassesment_link.csv"

# Load environment variables
try:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not PINECONE_API_KEY or not GOOGLE_API_KEY:
        raise ValueError("Missing required environment variables: PINECONE_API_KEY or GOOGLE_API_KEY")
except Exception as e:
    logger.error(f"Environment variable error: {e}")
    raise

# Load the assessment links DataFrame
try:
    assessment_links_df = pd.read_csv(ASSESSMENT_LINK_CSV_PATH)
    assessment_url_map = dict(zip(assessment_links_df['Assessment_Name'], assessment_links_df['Assessment_Link']))
    logger.info(f"Loaded {len(assessment_url_map)} assessment links from {ASSESSMENT_LINK_CSV_PATH}")
except Exception as e:
    logger.error(f"Failed to load assessment links CSV: {e}")
    raise

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    logger.info(f"Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

# Initialize embeddings
try:
    langchain_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info(f"HuggingFaceEmbeddings for {EMBEDDING_MODEL} initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}")
    raise

# Initialize Pinecone vector store
try:
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=langchain_embeddings,
        text_key="text"
    )
    logger.info("LangChain PineconeVectorStore initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PineconeVectorStore: {e}")
    raise

# Initialize Gemini for metadata extraction
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, api_key=GOOGLE_API_KEY)
    logger.info("Gemini model initialized successfully")
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

# Build the extraction chain
extract_chain = LLMChain(llm=llm, prompt=prompt_template_extract)

# Pydantic model for request validation
class QueryRequest(BaseModel):
    query: str

# Pydantic model for response structure
class AssessmentRecommendation(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: list[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: list[AssessmentRecommendation]

# Health Check Endpoint
@app.get("/health")
async def health():
    try:
        # Test Pinecone connection
        stats = pc.Index(INDEX_NAME).describe_index_stats()
        logger.info(f"Pinecone index stats: {stats}")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Recommendation Endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: QueryRequest):
    try:
        query = request.query
        logger.info(f"Received query: {query}")

        # Step 1: Extract metadata using Gemini
        logger.info("Invoking Gemini for metadata extraction")
        extraction_result = extract_chain.invoke({"query": query})
        logger.info(f"Raw extraction result: {extraction_result}")

        # Clean and parse the extracted metadata
        raw_text = extraction_result['text'].strip()
        logger.info(f"Raw text from Gemini: {raw_text}")
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            extracted_metadata = json.loads(json_match.group())
            logger.info(f"Parsed metadata: {extracted_metadata}")
        else:
            logger.warning(f"Failed to find valid JSON in: {raw_text}")
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

        logger.info(f"Extracted metadata - Skills: {skills}, Duration: {duration}, Languages: {languages}, Role Level: {role_level}, Test Type: {test_type}, Remote Testing: {remote_testing}, Adaptive Testing: {adaptive_testing}")

        # Step 2: Construct the query for Pinecone
        pinecone_query = f"assessment with skills {', '.join(skills)}"
        if role_level:
            pinecone_query += f" at {role_level} level"
        if test_type:
            pinecone_query += f" testing {', '.join(test_type)} skills"
        logger.info(f"Pinecone query: {pinecone_query}")

        # Step 3: Retrieve exactly 10 assessments from Pinecone
        logger.info("Performing Pinecone similarity search")
        results = vector_store.similarity_search(pinecone_query, k=10)
        logger.info(f"Retrieved {len(results)} assessments")

        if len(results) < 10:
            logger.warning(f"Only {len(results)} assessments retrieved, expected 10. Check query or index data.")

        # Step 4: Format the response
        recommended_assessments = []
        for result in results:
            metadata = result.metadata
            assessment_name = metadata.get("assessment_name", "Unknown Assessment")
            assessment_url = assessment_url_map.get(assessment_name, "https://www.shl.com/products/product-catalog/view/unknown/")
            recommended_assessments.append({
                "url": assessment_url,
                "adaptive_support": metadata.get("adaptive_testing", "No"),
                "description": metadata.get("description", "No description available"),
                "duration": int(metadata.get("duration", 0)),
                "remote_support": metadata.get("remote_testing", "No"),
                "test_type": metadata.get("test_type", [])
            })

        return {"recommended_assessments": recommended_assessments}

    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
