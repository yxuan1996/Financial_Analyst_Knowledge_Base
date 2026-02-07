import os
import json
import time
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
from supabase import create_client

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = "../data/raw"
USER_AGENT = os.environ["SEC_USER_AGENT"]
HEADERS = {"User-Agent": USER_AGENT}

FILING_TYPES = {"10-K", "10-Q"}
YEARS_BACK = 5
REQUEST_DELAY = 0.2

MAGNIFICENT_7 = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft Corp",
    "GOOGL": "Alphabet Inc",
    "AMZN": "Amazon.com Inc",
    "NVDA": "NVIDIA Corp",
    "META": "Meta Platforms Inc",
    "TSLA": "Tesla Inc",
}

# SEC URLs
CIK_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives"

# Supabase
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_BUCKET = os.environ["SUPABASE_BUCKET"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# HELPERS
# ----------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_cik_lookup():
    r = requests.get(CIK_LOOKUP_URL, headers=HEADERS)
    r.raise_for_status()
    return {
        item["ticker"]: str(item["cik_str"]).zfill(10)
        for item in r.json().values()
    }


def download_file(url, out_path):
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


def upload_to_supabase(local_path, storage_path):
    with open(local_path, "rb") as f:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path,
            file=f,
            file_options={"content-type": "application/octet-stream"},
        )


def get_quarter(date_str):
    month = int(date_str[5:7])
    return f"Q{((month - 1) // 3) + 1}"


# ----------------------------
# MAIN PIPELINE
# ----------------------------

def run():
    cik_lookup = get_cik_lookup()
    cutoff_date = datetime.now() - timedelta(days=365 * YEARS_BACK)

    for ticker, company in MAGNIFICENT_7.items():
        print(f"\n📄 {ticker} — {company}")

        cik = cik_lookup[ticker]
        company_dir = os.path.join(BASE_DIR, ticker)
        ensure_dir(company_dir)

        submissions = requests.get(
            SUBMISSIONS_URL.format(cik=cik),
            headers=HEADERS,
        ).json()

        filings = submissions["filings"]["recent"]

        for i in tqdm(range(len(filings["form"])), desc=ticker):
            form = filings["form"][i]
            if form not in FILING_TYPES:
                continue

            filing_date = datetime.strptime(filings["filingDate"][i], "%Y-%m-%d")
            if filing_date < cutoff_date:
                continue

            accession = filings["accessionNumber"][i].replace("-", "")
            primary_doc = filings["primaryDocument"][i]

            year = filing_date.year
            quarter = get_quarter(filings["filingDate"][i])

            if form == "10-K":
                base_name = f"{year}_10K"
            else:
                base_name = f"{year}_{quarter}_10Q"

            # --------------------
            # HTML
            # --------------------
            html_url = (
                f"{ARCHIVES_BASE}/edgar/data/"
                f"{int(cik)}/{accession}/{primary_doc}"
            )

            html_path = os.path.join(company_dir, f"{base_name}.html")

            print("Uploading HTML")
            print(html_path)

            if not os.path.exists(html_path):
                download_file(html_url, html_path)
                upload_to_supabase(
                    html_path,
                    f"{ticker}/{base_name}.html"
                )

            # --------------------
            # PDF
            # --------------------
            index_url = (
                f"{ARCHIVES_BASE}/edgar/data/"
                f"{int(cik)}/{accession}/{accession}-index.html"
            )

            pdf_url = index_url.replace("-index.html", ".pdf")
            pdf_path = os.path.join(company_dir, f"{base_name}.pdf")

            try:
                print("Uploading PDF")
                print(pdf_path)
                if not os.path.exists(pdf_path):
                    download_file(pdf_url, pdf_path)
                    upload_to_supabase(
                        pdf_path,
                        f"{ticker}/{base_name}.pdf"
                    )
            except Exception:
                # Some filings legitimately have no PDF
                pass

            time.sleep(REQUEST_DELAY)

        print(f"✅ Completed {ticker}")


if __name__ == "__main__":
    run()
