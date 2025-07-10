import requests
import json
import urllib3
import os
import zipfile
import base64
from datetime import datetime, timedelta
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_and_save_base64_parallel(auth_token, document_ids, output_folder, max_workers=8):
    url_template = "https://dms.rdc.in/rest/documents/download/{}"
    headers = {"Authorization": auth_token}
    os.makedirs(output_folder, exist_ok=True)

    def download_and_save(document_id):
        try:
            url = url_template.format(document_id)
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            if response.status_code == 200:
                base64_content = response.text.strip()
                if not base64_content:
                    return f"⚠️ {document_id} - Empty content"
                file_path = os.path.join(output_folder, f"{document_id}.txt")
                with open(file_path, "w") as f:
                    f.write(base64_content)
                return f"✅ {document_id}"
            else:
                return f"❌ {document_id} - Status {response.status_code}"
        except Exception as e:
            return f"❌ {document_id} - Error: {e}"

    print(f"\n🚀 Downloading {len(document_ids)} documents using {max_workers} threads...")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_and_save, doc_id): doc_id for doc_id in document_ids}
        for future in as_completed(futures):
            result = future.result()
            print(result)
            results.append(result)
    return results

def decode_base64_txts_to_pdf(txt_folder, pdf_folder, log_file="decode_log.txt"):
    log_entries = []
    os.makedirs(pdf_folder, exist_ok=True)

    all_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]

    for filename in all_files:
        txt_path = os.path.join(txt_folder, filename)
        try:
            with open(txt_path, "r") as f:
                content = f.read()

            # Extract base64 from JSON
            try:
                data = json.loads(content)
                base64_str = data.get("documentFile", "").strip()
            except Exception as parse_err:
                log_entries.append(f"{filename}: JSON parse error ({parse_err})")
                continue

            if not base64_str:
                log_entries.append(f"{filename}: 'documentFile' missing or empty")
                continue

            try:
                pdf_bytes = base64.b64decode(base64_str)
            except Exception as decode_err:
                log_entries.append(f"{filename}: base64 decode error ({decode_err})")
                continue

            pdf_filename = os.path.splitext(filename)[0] + ".pdf"
            pdf_path = os.path.join(pdf_folder, pdf_filename)
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(pdf_bytes)
            print(f"📄 PDF saved: {pdf_path}")

        except Exception as e:
            log_entries.append(f"{filename}: general error ({e})")

    if log_entries:
        log_path = os.path.join(pdf_folder, log_file)
        with open(log_path, "w") as f:
            f.write("\n".join(log_entries))
        print(f"\n⚠️ Issues encountered. Log saved at: {log_path}")

def main():
    auth_token = input("Enter Authorization Token: ").strip()
    month_year_str = input("Enter Month and Year (e.g., Mar-2025): ").strip()
    plant_code = input("Enter PLANT_CODE (e.g., DE1): ").strip()

    try:
        month_year = datetime.strptime(month_year_str, "%b-%Y")
    except ValueError:
        print("❌ Invalid format. Use MMM-YYYY (e.g., Mar-2025)")
        return

    year = month_year.year
    month = month_year.month
    start_date = datetime(year, month, 1)
    end_date = datetime(year, month, calendar.monthrange(year, month)[1])

    url = "https://dms.rdc.in/rest/documentclasses/105/indexsearch"
    headers = {"Authorization": auth_token, "Content-Type": "application/json"}

    base64_folder = "base64_txts"
    pdf_folder = "decoded_pdfs"
    os.makedirs(base64_folder, exist_ok=True)

    all_document_ids = []
    current_date = start_date

    print(f"\n🔍 Searching documents from {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}")

    while current_date <= end_date:
        in_date = current_date.strftime("%d-%b-%Y")
        payload = {
            "documentIndexes": [
                {"indexName": "IN_DATE", "value1": in_date, "operator": 1},
                {"indexName": "PLANT_CODE", "value1": plant_code, "operator": 1}
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, verify=False)
            response.raise_for_status()
            data = response.json()
            found_ids = [doc.get("documentId") for doc in data.get("documents", []) if doc.get("documentId")]
            all_document_ids.extend(found_ids)
            print(f"{in_date}: ✅ {len(found_ids)} documents")
        except Exception as e:
            print(f"{in_date}: ❌ Request error - {e}")

        current_date += timedelta(days=1)

    if not all_document_ids:
        print("📭 No document IDs found.")
        return

    print(f"\n📄 Total document IDs: {len(all_document_ids)}")

    # Step 1: Parallel download base64 .txts
    fetch_and_save_base64_parallel(auth_token, all_document_ids, base64_folder)

    # Step 2: Decode base64 to PDF from JSON
    decode_base64_txts_to_pdf(base64_folder, pdf_folder)

    # Step 3: Zip the PDFs
    zip_path = "decoded_pdfs.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for filename in os.listdir(pdf_folder):
            file_path = os.path.join(pdf_folder, filename)
            zipf.write(file_path, arcname=filename)

    print(f"\n✅ All PDFs zipped as: {zip_path}")

if __name__ == "__main__":
    main()
