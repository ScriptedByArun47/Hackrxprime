# app/extractor.py

import requests
import mimetypes
import fitz  # PyMuPDF
import docx
import email
from email import policy
from bs4 import BeautifulSoup
from io import BytesIO
from transformers import AutoTokenizer
import re

# Load tokenizer once (512-token limit for Gemini/Mistral)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# --- 📄 File Extractors ---

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_eml(file_bytes: bytes) -> str:
    msg = email.message_from_bytes(file_bytes, policy=policy.default)
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                return part.get_content()
            elif ctype == "text/html":
                return BeautifulSoup(part.get_content(), "html.parser").get_text()
    return msg.get_content()

# --- ✂️ Clause Splitter ---

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).replace('\xa0', ' ').strip()

def split_into_clauses(text: str):
    text = text.replace('\r', '').replace('\xa0', ' ').strip()

    # Sentence-level splitting
    sentences = re.split(r'(?<=[.!?]) +', text)
    buffer = ''
    blocks = []
    for sentence in sentences:
        candidate = buffer + ' ' + sentence if buffer else sentence
        if len(tokenizer.tokenize(candidate)) <= 512:
            buffer = candidate
        else:
            if buffer:
                blocks.append(buffer.strip())
            buffer = sentence
    if buffer:
        blocks.append(buffer.strip())

    # Preserve section headers (optional)
    header_pattern = re.compile(r'^(Section\s+\d+|[0-9.]{1,5}\s+[\w ]{3,})')
    combined_blocks = []
    current_header = ''

    for block in blocks:
        if header_pattern.match(block.strip()):
            current_header = block.strip()
        else:
            full_block = f"{current_header}\n{block.strip()}" if current_header else block.strip()
            combined_blocks.append(full_block)
            current_header = ''

    # Final chunking with tokenizer (optional re-chunk if >512 tokens)
    final = []
    for idx, block in enumerate(combined_blocks):
        tokens = tokenizer.tokenize(clean_text(block))
        if len(tokens) <= 512:
            final.append({
                "clause": clean_text(block),
                "id": f"clause_{len(final)+1}"
            })
        else:
            for i in range(0, len(tokens), 512):
                chunk = tokens[i:i + 512]
                text_chunk = tokenizer.convert_tokens_to_string(chunk)
                final.append({
                    "clause": clean_text(text_chunk),
                    "id": f"clause_{len(final)+1}"
                })

    return final

# --- 🌐 Entry Point ---

import hashlib
import os

def extract_clauses_from_url(url: str):
    response = requests.get(url)
    file_bytes = response.content
    content_type = response.headers.get("Content-Type")
    mime_type, _ = mimetypes.guess_type(url)

    if not mime_type and content_type:
        mime_type = content_type

    # Step 1: Extract text
    if mime_type:
        if "pdf" in mime_type:
            raw_text = extract_text_from_pdf(file_bytes)
        elif "word" in mime_type or "docx" in mime_type:
            raw_text = extract_text_from_docx(file_bytes)
        elif "plain" in mime_type or url.endswith(".txt"):
            raw_text = extract_text_from_txt(file_bytes)
        elif "message/rfc822" in mime_type or url.endswith(".eml"):
            raw_text = extract_text_from_eml(file_bytes)
        else:
            raw_text = extract_text_from_pdf(file_bytes)
    else:
        raw_text = extract_text_from_pdf(file_bytes)

    # Step 2: Split into clauses first
    clauses = split_into_clauses(raw_text)

    # Step 3: Heuristically check if it's an insurance policy
    policy_keywords = {
        "policy", "insurance", "sum insured", "coverage", "benefit",
        "premium", "claim", "hospitalization", "waiting period", "pre-existing"
    }
    match_count = 0
    for clause in clauses[:40]:  # Only scan first few to avoid false positives
        text = clause.get("clause", "").lower()
        if sum(1 for kw in policy_keywords if kw in text) >= 2:
            match_count += 1

    if match_count < 5:
        print(f"❌ Skipped non-policy document (matched {match_count} clauses):", url)
        # Delete stale clause cache
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join("clause_cache", f"{cache_key}.json")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"🧹 Removed existing clause cache: {cache_path}")
        return []

    return clauses  # ✅ Only return the clauses once




