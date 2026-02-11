from pathlib import Path
import sys, pymupdf  # import the bindings
import pymupdf4llm
import tiktoken
import os
from openai import AzureOpenAI
import json
import yaml
from neo4j import GraphDatabase
import time
from typing import Dict, Any, List
from tqdm import tqdm
import uuid
from pinecone import Pinecone
import re

from dotenv import load_dotenv

import sentry_sdk

# Load environment variables from .env file
load_dotenv() 

# ----------------------------
# CONFIG
# ----------------------------
# 1. Configuration - Replace with your Azure OpenAI details
ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
GPT_4_DEPLOYMENT = "gpt-4o"
LLAMA_4_DEPLOYMENT = "Llama-4-Scout-17B-16E-Instruct"
API_VERSION = "2024-12-01-preview" # Or the latest stable version

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT
)

sentry_sdk.init(
    dsn=os.environ["SENTRY_DSN"],
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
)


# 1. Define your paths
source_dir = Path("../data/pdf")
output_dir = Path("../data/pdf_processed")
output_dir_json = Path("../data/pdf_json")

# 2. Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)
output_dir_json.mkdir(parents=True, exist_ok=True)

# 3. Helper functions
## 3.1 Chunking, Cleaning mojibake from markdown 
enc = tiktoken.get_encoding("cl100k_base")

def chunk_by_paragraphs(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 100,
) -> list[str]:
    """
    Chunk text by paragraphs with token limits and overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(enc.encode(para))

        if current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current))

            # build overlap
            overlap = []
            overlap_count = 0
            for p in reversed(current):
                t = len(enc.encode(p))
                if overlap_count + t > overlap_tokens:
                    break
                overlap.insert(0, p)
                overlap_count += t

            current = overlap + [para]
            current_tokens = overlap_count + para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def clean_mojibake_markdown(markdown_text: str) -> str:

    prompt = f"""
        You are a text normalization system.

        Task:
        - Fix mojibake and encoding issues
        - Preserve ALL content
        - Preserve markdown structure
        - Do NOT summarize, truncate, or rewrite
        - Output the corrected markdown only

        Text:
        \"\"\"
        {chunk}
        \"\"\"
    """

    response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


## 3.2 Vector Embedding related functions
def extract_vector_metadata(
    markdown_text: str,
    company_ticker: str,
    fiscal_year: int,
    document_type: str,
    json_schema: dict,
    allowed_values: dict,
) -> dict:


    prompt = f"""
    You are an ontology-guided classification system.

    Your task:
    1. Fix any encoding and formating issues in the text
        - Fix mojibake and encoding issues
        - Preserve ALL content
        - Preserve markdown structure
        - Do NOT summarize, truncate, or rewrite
    2. Identify ontology elements explicitly mentioned
    3. Select ONLY from allowed values
    4. Return ONLY valid JSON matching the schema exactly

    Allowed values (authoritative):
    {json.dumps(allowed_values, indent=2)}

    Rules:
    - Do NOT invent metrics, roles, or document types
    - Do NOT use synonyms
    - Use empty arrays if nothing is found
    - Confidence values must be between 0 and 1
    - Output JSON ONLY (no markdown, no explanation)

    JSON schema (structure must match):
    {json.dumps(json_schema, indent=2)}

    Input markdown:
    \"\"\"
    {markdown_text}
    \"\"\"
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    result = json.loads(response.choices[0].message.content)

    # Inject authoritative metadata (prevent drift)
    result["company_ticker"] = company_ticker
    result["fiscal_year"] = fiscal_year
    result["document_type"] = document_type
    # result["text"] = markdown_text

    return result


def generate_embeddings_from_json(
    records: List[Dict],
    batch_size: int = 100,
    delay_seconds: float = 2.5,
) -> List[List[float]]:
    """
    Generate embeddings in batches with rate limiting and progress display.

    Args:
        records: List of JSON objects, each containing a 'text' key
        batch_size: Number of texts per embedding request
        delay_seconds: Delay between batches (seconds)

    Returns:
        List of embedding vectors aligned with input order
    """

    if not records:
        return []

    texts = []
    for i, record in enumerate(records):
        if "text" not in record:
            raise ValueError(f"Record at index {i} is missing 'text' field")
        texts.append(record["text"])

    embeddings: List[List[float]] = []

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches), desc="Generating embeddings"):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch_texts,
        )

        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

        # Rate limiting delay (skip delay after last batch)
        if batch_idx < total_batches - 1:
            time.sleep(delay_seconds)

    return embeddings


def insert_into_pinecone(
    embeddings: List[List[float]],
    metadata_list: List[Dict],
    batch_size: int = 100,
):
    """
    Insert embeddings with metadata into a Pinecone index.

    Args:
        embeddings: List of embedding vectors
        metadata_list: List of metadata dicts (same order as embeddings)
        batch_size: Number of vectors per upsert batch
    """

    if len(embeddings) != len(metadata_list):
        raise ValueError("Embeddings and metadata list must be the same length")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_HOST"))

    vectors = []

    for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
        vector_id = metadata["company_ticker"] + "_" + str(metadata["fiscal_year"]) + "_" + metadata['document_type'] + "_" + str(i)

        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata,
        })

    # Batch upsert
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    print("Insert embeddings into vector DB complete!")

def flatten_metadata(metadata: Any, parent_key="", sep=".") -> Dict:
    items = []
    
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            # Recursive call for both dicts and lists
            if isinstance(v, (dict, list)):
                items.extend(flatten_metadata(v, new_key, sep).items())
            else:
                items.append((new_key, v))
                
    elif isinstance(metadata, list):
        for i, v in enumerate(metadata):
            # Create a key like "mentions.metrics.0.Revenue"
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (dict, list)):
                items.extend(flatten_metadata(v, new_key, sep).items())
            else:
                items.append((new_key, v))
                
    return dict(items)

def sanitize_for_pinecone(flat_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures all values are Pinecone-compatible types."""
    sanitized = {}
    
    # Pinecone supported types
    allowed_types = (str, int, float, bool)
    
    for k, v in flat_metadata.items():
        # 1. Handle Nulls/None
        if v is None:
            sanitized[k] = 0
            
        # 2. Handle supported types
        elif isinstance(v, allowed_types):
            sanitized[k] = v
            
        # 3. Handle simple lists of strings (Pinecone allows these)
        elif isinstance(v, list) and all(isinstance(item, str) for item in v):
            sanitized[k] = v
            
        # 4. Fallback for everything else (complex objects, empty lists, etc.)
        else:
            sanitized[k] = 0
            
    return sanitized


## 3.4 Image processing related functions

def convert_pdf_to_image(fname, subfolder, year_prefix):
    doc = pymupdf.open(fname)  # open document
    for page in doc:  # iterate through the pages
        pix = page.get_pixmap()  # render page to an image
        output_path = str(output_dir) + f"/{subfolder}/{year_prefix}/{page.number:03}.png"
        # print(output_path)
        pix.save(output_path)  # store image as a PNG


# 4. Main running code

for pdf_path in source_dir.rglob("*.pdf"):
    # Extract subdirectory name
    subfolder = pdf_path.parent.name

    # Extract the year prefix (assuming format "2023_filename.pdf")
    # .stem gets the filename without the extension
    filename_parts = pdf_path.stem.split('_')
    year_prefix = filename_parts[0]

    # print(subfolder)
    # print(year_prefix)
    print(pdf_path)

    # Extract Text
    md_text = pymupdf4llm.to_markdown(pdf_path)

    # Chunk and Generate Vector Embeddings
    chunks = chunk_by_paragraphs(md_text)

    with open('json_schema.json', 'r') as file:
        json_schema = json.load(file)

    with open('allowed_values.json', 'r') as file:
        allowed_values = json.load(file)

    # # Clean Mojibake, generate json metadata
    clean_chunks = []
    for chunk in tqdm(clean_chunks, desc="Generating json metadata"):

        try:

            chunk_json = extract_vector_metadata(
            chunk,
            subfolder,
            int(year_prefix),
            "ARS",
            json_schema,
            allowed_values,
            )

            clean_chunks.append(chunk_json)

        except Exception as e:
            print(e)

    print("Metadata generation complete")

    

    # Generate Vector Embeddings
    embeddings = generate_embeddings_from_json(
        clean_chunks,
        50,
        2.5,
    ) 

    print("Vectors generated")

    # # Flatten Metadata
    flatten_chunks = []
    for i in clean_chunks:
        flatten_chunks.append(flatten_metadata(i))

    final_metadata = []
    for i in flatten_chunks:
        final_metadata.append(sanitize_for_pinecone(i))
    
    # Insert into Vector DB
    try:
        insert_into_pinecone(
            embeddings=embeddings,
            metadata_list=final_metadata,
            batch_size=50,
        )
        print("Pinecone insertion complete")
    except Exception as e:
        print("pinecone insertion failed")
        print(e)

    # Save metadata to json for later reference
    filename = str(output_dir_json) + f"/{subfolder}_{year_prefix}.json"

    # Open the file and dump the data
    try:
        with open(filename, 'w') as json_file:
            json.dump(final_metadata, json_file, indent=4)
        print(f"Successfully saved data to {filename}")
    except IOError as e:
        print(f"Error saving file: {e}")

 





