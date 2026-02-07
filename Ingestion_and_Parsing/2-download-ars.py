import os
import requests
from datetime import datetime
from tqdm import tqdm
from supabase import create_client

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = "../data/pdf"
YEARS_BACK = 5

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_BUCKET = os.environ["SUPABASE_BUCKET_PDF"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

HEADERS = {
    "User-Agent": "RAG-Research your@email.com",
    "Accept-Encoding": "gzip, deflate",
}

# Note: Apple does not typically release a glossy annual report to shareholders. Form 10-K serves as their annual report. 

MAG7 = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "NVDA": "0001045810",
    "META": "0001326801",
    "TSLA": "0001318605",
}

# ----------------------------
# HELPERS
# ----------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def upload_to_supabase(local_path, storage_path):
    with open(local_path, "rb") as f:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            storage_path,
            f,
            file_options={"content-type": "application/pdf"},
        )


def get_submissions(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def find_ars_filings(submissions):
    filings = submissions["filings"]["recent"]
    results = []

    for i, form in enumerate(filings["form"]):
        if form in ("ARS", "ARS/A"):
            results.append({
                "accession": filings["accessionNumber"][i].replace("-", ""),
                "date": filings["filingDate"][i],
            })

    return results


def find_pdf_document(cik, accession):
    index_url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/{cik}.json"
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/"
    r = requests.get(base + "index.json", headers=HEADERS, timeout=30)
    r.raise_for_status()

    for file in r.json()["directory"]["item"]:
        if file["name"].lower().endswith(".pdf"):
            return base + file["name"], file["name"]

    return None, None


# ----------------------------
# MAIN PIPELINE
# ----------------------------

def run():
    current_year = datetime.now().year

    for ticker, cik in MAG7.items():
        print(f"\n📄 Processing {ticker}")
        ticker_dir = os.path.join(BASE_DIR, ticker)
        ensure_dir(ticker_dir)

        submissions = get_submissions(cik)
        ars_filings = find_ars_filings(submissions)

        for filing in tqdm(ars_filings):
            year = int(filing["date"][:4])
            if year < current_year - YEARS_BACK:
                continue

            pdf_url, original_name = find_pdf_document(cik, filing["accession"])
            if not pdf_url:
                continue

            filename = f"{year}_{original_name}"
            local_path = os.path.join(ticker_dir, filename)
            storage_path = f"pdf/{ticker}/{filename}"

            if os.path.exists(local_path):
                continue

            try:
                r = requests.get(pdf_url, headers=HEADERS, timeout=30)
                r.raise_for_status()

                with open(local_path, "wb") as f:
                    f.write(r.content)

                upload_to_supabase(local_path, storage_path)

            except Exception as e:
                print(f"⚠️ Failed {ticker} {year}: {e}")

        print(f"✅ Finished {ticker}")


if __name__ == "__main__":
    run()
