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
from typing import List, Dict
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

# 2. Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

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
    1. Keep the provided markdown text unchanged
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
    result["text"] = markdown_text

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

def flatten_metadata(metadata: Dict, parent_key="", sep=".") -> Dict:
    items = []
    for k, v in metadata.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_metadata(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


## 3.3 GraphDB related functions

def generate_cypher_constraints(ontology_yaml: str) -> list[str]:

    prompt = f"""
You are a Neo4j data modeler.

Given the following ontology schema in YAML, generate Cypher CONSTRAINT statements only.

Rules:
- Use the latest Neo4j 5.x syntax
- Use IF NOT EXISTS
- ON and ASSERT SHOULD NOT be used. Replace ON with FOR and ASSERT with REQUIRE. 
- Enforce uniqueness where appropriate
- Do NOT include comments
- Output one Cypher statement per line

Ontology:
{ontology_yaml}
"""

    response = client.chat.completions.create(
        model=GPT_4_DEPLOYMENT,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    cypher = response.choices[0].message.content.strip()
    return [line.strip() for line in cypher.split("\n") if line.strip()]


def generate_ingestion_cypher(
    ontology_yaml: str,
    markdown_text: str,
    company_ticker: str,
    fiscal_year: int,
    document_type: str
) -> list[str]:

    prompt = f"""
You are an information extraction system that generates Neo4j Cypher.

Ontology (authoritative):
{ontology_yaml}

Document metadata:
- company_ticker: {company_ticker}
- fiscal_year: {fiscal_year}
- document_type: {document_type}

Extract entities and relationships from the markdown text and generate Cypher queries.

Rules:
- Use the latest Neo4j 5.x syntax
- Use MERGE, not CREATE
- DO NOT use MATCH
- Do NOT invent metrics outside ontology enums
- Use canonical metric names only
- Create [Document, Company, FiscalYear, Metric, KeyPerson, KeyDevelopment] nodes as needed. 
- Document id and title should be a combination of company_ticker, fiscal_year and document_type
- Output ONLY Cypher statements
- One statement per line
- Only 1 node is allowed to be created per statement. If creating multiple nodes, use multiple statements.
- Only 1 relationship is allowed to be created per statement. If creating multiple relationships, use multiple statements.
- Each Cypher statement MUST end with a semicolon (;)
- Ensure that nodes are wrapped in Parentheses ()
- Node properties should be contained within curly braces as a key-value pair 

Markdown text:
{markdown_text}
"""

    response = client.chat.completions.create(
        model=LLAMA_4_DEPLOYMENT,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    cypher = response.choices[0].message.content.strip()
    # return [line.strip() for line in cypher.split("\n") if line.strip()]
    return cypher

def llm_correct_cypher(
    cypher_query: str,
    ontology_yaml: str,
    error_message: str | None = None,
) -> List[str]:
    """
    Uses GPT-4o to validate and correct Cypher queries.
    May split a single query into multiple valid Cypher queries.
    Returns a list of Cypher query strings.
    """

    system_prompt = """
You are a Neo4j Cypher expert. Fix the given queries as needed. 

Rules:
- Use ONLY labels, relationships, and properties defined in the ontology.
- Relationships should only exist between documents and other entities. 
- Do NOT invent new schema elements.
- Preserve the original semantic intent.
- DO NOT use MATCH.
- use MERGE.
- If necessary, split the query into multiple Cypher queries.
- Each query must be executable independently.
- Remove duplicate queries if you see them. 
- Output MUST be valid JSON matching this schema:

{
  "queries": ["string"]
}

Do not include explanations, comments, or markdown.
"""

    user_prompt = f"""
Ontology schema (YAML):
{ontology_yaml}

Original Cypher query:
{cypher_query}
"""

    if error_message:
        user_prompt += f"""

Neo4j error message:
{error_message}

Fix the Cypher query to resolve this error.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
        queries = parsed.get("queries", [])
        if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
            raise ValueError("Invalid query list format")
        return [q.strip() for q in queries if q.strip()]

    except Exception as e:
        raise ValueError(
            f"LLM returned invalid JSON.\nContent:\n{content}\nError: {e}"
        )


def combine_cypher_statements(statements: list[str]) -> str:
    combined = "\n".join(
        stmt.rstrip(";") for stmt in statements
    )
    return combined + ";"



# Neo4j community edition only allows for a single default database
def execute_cypher(
    cypher_statements: list[str],
    neo4j_uri: str,
    username: str,
    password: str
):

    combined_query = combine_cypher_statements(cypher_statements)

    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(username, password)
    )

    with driver.session() as session:
        # for stmt in cypher_statements:
        #     session.run(stmt)
        session.run(combined_query)

    driver.close()


def validate_and_execute_cypher(
    cypher_queries,
    ontology_yaml,
    max_retries=3,
    retry_delay=1.5,
):



    attempt = 0
    current_queries = cypher_queries

    while attempt < max_retries:
        try:
            execute_cypher(
            current_queries,
            neo4j_uri=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"]
            )

            print("✓ Queries executed successfully")
            break

        except Exception as e:
            attempt += 1
            error_msg = str(e)
            print(f"✗ Execution failed (attempt {attempt})")
            print(error_msg)

            if attempt >= max_retries:
                print("⚠ Max retries reached. Skipping.")
                break

            # Feed the *failing* combined query back to LLM
            combined_query = "\n".join(current_queries)

            current_queries = llm_correct_cypher(
                cypher_query=combined_query,
                ontology_yaml=ontology_yaml,
                error_message=error_msg,
            )

            time.sleep(retry_delay)

def load_cypher_queries(file_path: str) -> list[str]:
    """
    Reads a text file containing Cypher queries separated by two empty lines
    and returns a list of Cypher query strings.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on two or more blank lines (handles spaces/tabs safely)
    queries = re.split(r"\n\s*\n\s*\n+", content)

    # Clean up each query
    queries = [
        q.strip()
        for q in queries
        if q.strip()
    ]

    return queries

## 3.4 Image processing related functions

def convert_pdf_to_image(fname, subfolder, year_prefix):
    doc = pymupdf.open(fname)  # open document
    for page in doc:  # iterate through the pages
        pix = page.get_pixmap()  # render page to an image
        output_path = str(output_dir) + f"/{subfolder}/{year_prefix}/{page.number:03}.png"
        # print(output_path)
        pix.save(output_path)  # store image as a PNG


# 4. Main running code
with open("financial_ontology.yaml") as f:
    ontology_yaml = f.read()

# constraints = generate_cypher_constraints(ontology_yaml)
# constraints.remove("```cypher")
# constraints.remove("```")
# print(constraints)

# try:
#     execute_cypher(
#     constraints,
#     neo4j_uri=os.environ["NEO4J_URI"],
#     username=os.environ["NEO4J_USERNAME"],
#     password=os.environ["NEO4J_PASSWORD"]
#     )
# except Exception as e:
#     print("Cypher constraint creation error")
#     print(e)


# Iterate through all .pdf files recursively
for pdf_path in source_dir.rglob("*.pdf"):
    # Extract subdirectory name
    subfolder = pdf_path.parent.name

    # Extract the year prefix (assuming format "2023_filename.pdf")
    # .stem gets the filename without the extension
    filename_parts = pdf_path.stem.split('_')
    year_prefix = filename_parts[0]

    # print(subfolder)
    # print(year_prefix)
    # print(pdf_path)

    # Extract Text
    md_text = pymupdf4llm.to_markdown(pdf_path)

    # Populate Knowledge Graph
    cypher_ingest = generate_ingestion_cypher(
    ontology_yaml=ontology_yaml,
    markdown_text=md_text,
    company_ticker=subfolder,
    fiscal_year=int(year_prefix),
    document_type="ARS"
    )

    # Fix multiline cypher problem
    cypher_ingest_clean = []
    holding_var = ""
    for line in cypher_ingest:
        holding_var = holding_var + line
        if line.endswith(";"):
            cypher_ingest_clean.append(holding_var)
            # reset
            holding_var = ""

    file_path = "cypher_debug.txt"

    with open(file_path, 'w') as file:
        for item in cypher_ingest_clean:
            file.write(f"{item}\n")

    # cypher_ingest_clean = load_cypher_queries("cypher_debug.txt")

    cypher_ingest_clean = llm_correct_cypher(
        cypher_query=cypher_ingest_clean,
        ontology_yaml=ontology_yaml,
        error_message=None,
    )

    with open("cypher_corrected.txt", 'w') as file:
        for item in cypher_ingest_clean:
            file.write(f"{item}\n")
    

    validate_and_execute_cypher(
        cypher_ingest_clean,
        ontology_yaml,
        max_retries=3,
        retry_delay=1.5,
    )



# for pdf_path in source_dir.rglob("*.pdf"):
#     # Extract subdirectory name
#     subfolder = pdf_path.parent.name

#     # Extract the year prefix (assuming format "2023_filename.pdf")
#     # .stem gets the filename without the extension
#     filename_parts = pdf_path.stem.split('_')
#     year_prefix = filename_parts[0]

    # print(subfolder)
    # print(year_prefix)
    # print(pdf_path)

    # Extract Text
    # md_text = pymupdf4llm.to_markdown(pdf_path)

    # Chunk and Generate Vector Embeddings
    # chunks = chunk_by_paragraphs(md_text)

    # with open('json_schema.json', 'r') as file:
    #     json_schema = json.load(file)

    # with open('allowed_values.json', 'r') as file:
    #     allowed_values = json.load(file)

    # # Clean Mojibake, generate json metadata
    # clean_chunks = []
    # for chunk in chunks:
    #     clean_chunk = clean_mojibake_markdown(chunk)

    #     chunk_json = extract_vector_metadata(
    #     clean_chunk,
    #     subfolder,
    #     int(year_prefix),
    #     "ARS",
    #     json_schema,
    #     allowed_values,
    #     )

    #     clean_chunks.append(chunk_json)

    

    # # Generate Vector Embeddings
    # embeddings = generate_embeddings_from_json(
    #     clean_chunks,
    #     50,
    #     2.5,
    # ) 

    # # Flatten Metadata
    # flatten_chunks = flatten_metadata(clean_chunks)
    
    # # Insert into Vector DB
    # insert_into_pinecone(
    #     embeddings=embeddings,
    #     metadata_list=flatten_chunks,
    #     batch_size=50,
    # )





    # convert_pdf_to_image(pdf_path, subfolder, year_prefix)

    # Upload images to supabase

    # 



