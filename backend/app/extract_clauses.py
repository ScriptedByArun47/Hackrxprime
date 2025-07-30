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

# --- 📎 Clause Splitter ---

def split_into_clauses(text: str):
    text = text.replace('\r', '').replace('\xa0', ' ').strip()
    blocks = [b.strip() for b in text.split('\n\n') if len(b.strip()) > 40]

    if len(blocks) < 5:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        buffer = ''
        blocks = []
        for line in lines:
            buffer += ' ' + line
            if len(buffer.split()) >= 50:
                blocks.append(buffer.strip())
                buffer = ''
        if buffer:
            blocks.append(buffer.strip())

    final = []
    for block in blocks:
        tokens = tokenizer.tokenize(block)
        if len(tokens) <= 512:
            final.append({"clause": block})
        else:
            for i in range(0, len(tokens), 512):
                chunk = tokens[i:i + 512]
                text_chunk = tokenizer.convert_tokens_to_string(chunk)
                final.append({"clause": text_chunk})

    return final

# --- 🌐 Entry Point ---

def extract_clauses_from_url(url: str):
    response = requests.get(url)
    file_bytes = response.content
    content_type = response.headers.get("Content-Type")
    mime_type, _ = mimetypes.guess_type(url)

    if not mime_type and content_type:
        mime_type = content_type

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

    return split_into_clauses(raw_text)
