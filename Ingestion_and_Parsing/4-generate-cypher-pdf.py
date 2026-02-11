from pathlib import Path
import sys
import os
from openai import AzureOpenAI
import json
import yaml
from neo4j import GraphDatabase
import time
from typing import Dict, Any, List
from tqdm import tqdm
import uuid
import re

from dotenv import load_dotenv

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

# 1. Define your paths
source_dir = Path("../data/pdf")
output_dir = Path("../data/pdf_processed")
output_dir_json = Path("../data/pdf_json")

failed_output_dir = Path("../data/pdf_cypher_errors")

# 2. Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)
output_dir_json.mkdir(parents=True, exist_ok=True)
failed_output_dir.mkdir(parents=True, exist_ok=True)

def generate_ingestion_cypher(
    cypher_template: str,
    markdown_text: str,
    company_ticker: str,
    fiscal_year: int,
    document_type: str
) -> list[str]:

    prompt = f"""
You are an information extraction system that generates Neo4j Cypher.

Cypher template (authoritative):
{cypher_template}

Document metadata:
- company_ticker: {company_ticker}
- fiscal_year: {fiscal_year}
- document_type: {document_type}

Extract entities and relationships from the markdown text and generate Cypher queries. If there is no relevant information return an empty list.

Rules:
- Use the latest Neo4j 5.x syntax
- Use MERGE, not CREATE
- DO NOT use MATCH
- Follow the cypher template. Do not invent statements that do not conform to the template
- Use canonical metric names only
- Create [Document, Company, FiscalYear, Metric, KeyPerson, KeyDevelopment] nodes as needed
- KeyDevelopment nodes should contain a title and a description that summarized key development and advancements highlighted in the text
- Document id and title should be a combination of company_ticker, fiscal_year and document_type
- Output ONLY Cypher statements
- One statement per line
- A variable name can be introduced only once; later references must reuse it, or use a different variable name
- Only 1 node is allowed to be created per statement. If creating multiple nodes, use multiple statements.
- Only 1 relationship is allowed to be created per statement. If creating multiple relationships, use multiple statements.
- Relationships should only exist between documents and other entities.
- Each Cypher statement MUST end with a semicolon (;)
- Ensure that nodes are wrapped in Parentheses ()
- Node properties should be contained within curly braces as a key-value pair

Markdown text:
{markdown_text}
"""

    try:
        response = client.chat.completions.create(
            model=GPT_4_DEPLOYMENT,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        cypher = response.choices[0].message.content.strip()
        # return [line.strip() for line in cypher.split("\n") if line.strip()]
        return cypher
    except Exception as e:
        print(e)
        return []

def llm_correct_cypher(
    cypher_query: str,
    cypher_template: str,
    error_message: str | None = None,
) -> List[str]:
    """
    Uses GPT-4o to validate and correct Cypher queries.
    May split a single query into multiple valid Cypher queries.
    Returns a list of Cypher query strings.
    """

    system_prompt = """
You are a Neo4j Cypher expert. Fix the given queries as needed.

Case New Node Conflict:
- If the error message states that it can not create a new node due to conflicts with existing unique nodes, identify whether that statement is trying to create a single new entity or a new relation.
- If trying to create a single new entity, remove the statement entirely.
- If trying to create a new relation, merge using one key identifier property only.
- For Company nodes use ticker
- For Document node use document_id
- For KeyPerson node use name

Rules:
- Use ONLY labels, relationships, and properties defined in the cypher template.
- Relationships should only exist between documents and other entities.
- Do NOT invent new schema elements.
- Preserve the original semantic intent.
- DO NOT use MATCH.
- use MERGE.
- If necessary, split the query into multiple Cypher queries.
- Each query must be executable independently.
- Remove duplicate queries if you see them.
- A variable name can be introduced only once; later references must reuse it, or use a different variable name
- Output MUST be valid JSON matching this schema:

{
  "queries": ["string"]
}

Do not include explanations, comments, or markdown.
"""

    user_prompt = f"""
Cypher Template:
{cypher_template}

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
        model=GPT_4_DEPLOYMENT,
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

    # combined_query = combine_cypher_statements(cypher_statements)

    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(username, password)
    )

    with driver.session() as session:
        for stmt in cypher_statements:
            session.run(stmt)
        # session.run(combined_query)

    driver.close()


def validate_and_execute_cypher(
    cypher_queries,
    cypher_template,
    max_retries=3,
    retry_delay=1.5,
    failed_output_dir=failed_output_dir,
    subfolder="",
    year_prefix="",
    count=""
    ):

    attempt = 0
    current_queries = cypher_queries

    while attempt < max_retries:
        try:
            execute_cypher(
            current_queries,
            neo4j_uri=os.environ['NEO4J_URI'],
            username=os.environ['NEO4J_USERNAME'],
            password=os.environ['NEO4J_PASSWORD']
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
                with open(f"{str(failed_output_dir)}/{subfolder}_{year_prefix}_{count}.txt", 'w') as file:
                  for item in current_queries:
                        file.write(f"{item}\n")
                break

            # Feed the *failing* combined query back to LLM
            combined_query = "\n".join(current_queries)

            current_queries = llm_correct_cypher(
                cypher_query=combined_query,
                cypher_template=cypher_template,
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


with open("financial_ontology.yaml") as f:
    ontology_yaml = f.read()


with open("financial_cypher_template.txt", 'r') as file:
    cypher_template = file.read()

ontology_yaml = yaml.safe_load(ontology_yaml)

constraints = generate_cypher_constraints(ontology_yaml)
constraints.remove("```cypher")
constraints.remove("```")
print(constraints)

try:
    execute_cypher(
    constraints,
    neo4j_uri=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
    )
except Exception as e:
    print("Cypher constraint creation error")
    print(e)


# Use the glob('*.json') method to find all files ending with .json
for file_path in output_dir_json.glob('*.json'):
    print(f"Processing file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Successfully loaded data from {file_path}")
            text_list = [item.get('text') for item in data]

            subfolder = file_path.parent.name

            # Extract the year prefix
            # .stem gets the filename without the extension
            filename_parts = file_path.stem.split('_')
            year_prefix = filename_parts[0]

            print(subfolder)
            print(year_prefix)

            count = 0

            for md_text in text_list:

                count += 1

                # Populate Knowledge Graph
                cypher_ingest = generate_ingestion_cypher(
                cypher_template=cypher_template,
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

                # with open(f"..data/cypher_debug_{subfolder}_{year_prefix}_{count}.txt", 'w') as file:
                #     for item in cypher_ingest_clean:
                #         file.write(f"{item}\n")

                cypher_ingest_clean = llm_correct_cypher(
                    cypher_query=cypher_ingest_clean,
                    cypher_template=cypher_template,
                    error_message=None,
                )

                # with open(f"..data/cypher/cypher_corrected_{subfolder}_{year_prefix}_{count}.txt", 'w') as file:
                #     for item in cypher_ingest_clean:
                #         file.write(f"{item}\n")


                validate_and_execute_cypher(
                    cypher_ingest_clean,
                    cypher_template,
                    max_retries=5,
                    retry_delay=1.5,
                    failed_output_dir=failed_output_dir,
                    subfolder=subfolder,
                    year_prefix=year_prefix,
                    count=count
                )

                time.sleep(5)



    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
