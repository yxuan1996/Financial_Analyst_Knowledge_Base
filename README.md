# Financial Analysis RAG Data Ingestion Test

Scripts to test RAG Data Ingestion with SEC filling data on Magnificent 7 stocks. 

Contains the following features:
- Document ingestion (PDF and HTML)
- Chunking and Embedding (VectorDB)
- Knowledge Graph with Entity-Relationship modelling (Graph DB)

## 📌 Supported Companies (Magnificent 7)

| Company | Ticker |
|------|------|
| Apple | AAPL |
| Microsoft | MSFT |
| Alphabet (Google) | GOOGL |
| Amazon | AMZN |
| Nvidia | NVDA |
| Meta | META |
| Tesla | TSLA |

---


## Script Summary

#### 1-download-sec-fillings.py
Downloads **SEC 10-K and 10-Q filings** for the **Magnificent 7 technology companies** for the last 5 years from the SEC Edgar website and uploads them to Supabase Storage. 

The files are saved to the `../data/raw` directory. 

Note: Only HTML SEC fillings were successfully retrieved. 

#### 2-download-ars.py
Downloads **PDF Annual Reports** for the **Magnificent 7 technology companies** for the last 3 years from the SEC Edgar website and uploads them to Supabase Storage. 

The files are saved to the `../data/pdf` directory.

#### 3-process-pdf.py
For each PDF annual report:
- Extract text using pymupdfllm
- Chunk the text by paragraph
- Pass each chunk to LLM to clean mojibake and generate json metadata
- Generate vector embeddings for each text chunk
- Insert embeddings into Pinecone vector DB together with json metadata

The cleaned text along with the json metadata are saved in `../data/pdf_json`

`json_schema.json` defines the expected json metadata schema, while `allowed_values.json` contains the list of allowed values. 

The contents of these 2 files are passed into the LLM to ensure consistency during the json metadata generation process. 

#### 4-generate-cypher-pdf.py
We start by defining an ontology for financial analysis in `financial_ontology.yaml`.
This will form the basis for our knowledge graph schema. 

Using this ontology, we generate `financial_cypher_template.txt` that is passed into the LLM to ensure consistency during the cypher generation process. 

The script will iterate through the text chunks that are saved in `../data/pdf_json` from the previous section:
- Generate cypher queries from the text chunks
- Ensure that the cypher queries follow consistent formatting and rules
- Attempt to execute the cypher queries to create new nodes and relationships. If errors occur, we pass the error message back to the LLM and attempt to fix those queries and try again. We limit the max number of retry attempts to 5. 
- Failed cypher queries are saved to `../data/pdf_cypher_errors`. This acts like a dead-letter queue, allowing us to inspect and debug the failed queries. 


---

## Learnings and Observation

### Use OCR based text extraction for superior performance. 
I have tested out various libraries for text extraction. Pymupdf and Kreuzberg are fast and work well. However these PDF text extractors have several issues:
- The extracted text contains a lot of mojibake. We need to chunk the text and pass them into LLM to clean the text. 
- Difficulty in chunking since there are no obvious section markers. 
- Images and tables are not extracted. 

Most modern parsing libraries such as `llamaparse` and `unstructured io` use agentic methods for best results. 

### Use API whenever possible
It turns out that parsing HTML SEC Fillings is a lot of work! 
- We need to first convert the html file to markdown (Pandoc). 
- We can also use beautifulsoup to extract tables from HTML. 
- Unfortunately the tables can be messy and misaligned, so we need to pass it to an LLM to clean it and generate metadata. 

The [EdgarTools](https://edgartools.readthedocs.io/en/latest/) and Datamule python libraries provide a fast and easy way to obtain SEC Fillings data, and I shall be using those for my next project. 



---

## Usage Instructions
Copy `.env.example` and configure your own `.env` variables. 

Run `pip install -r requirements.txt` 

Run the scripts in sequential order. 

## 📂 Folder Structure

The script uses a deterministic, reproducible directory layout:

```
data/
└── raw/
├── AAPL/
│ ├── 2020_10K.html
│ ├── 2020_10K.pdf
│ ├── 2021_Q1_10Q.html
│ ├── 2021_Q1_10Q.pdf
├── MSFT/
├── NVDA/
└── ...
```


The **same structure is preserved** when files are uploaded to Supabase Storage.

---



