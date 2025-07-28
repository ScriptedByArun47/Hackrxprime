import requests
import mimetypes
import fitz  # PyMuPDF
import docx
import email
from email import policy
from bs4 import BeautifulSoup
from io import BytesIO

# --- File extractors ---

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file_bytes):
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_txt(file_bytes):
    return file_bytes.decode('utf-8', errors='ignore')

def extract_text_from_eml(file_bytes):
    msg = email.message_from_bytes(file_bytes, policy=policy.default)
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                return part.get_content()
            elif content_type == 'text/html':
                html = part.get_content()
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
    else:
        return msg.get_content()

# --- Clause splitter ---

def split_into_clauses(text: str):
    # Normalize and clean input
    text = text.replace('\r', '').replace('\xa0', ' ').strip()
    blocks = [b.strip() for b in text.split('\n\n') if len(b.strip()) > 40]

    # Fallback: reassemble line-based blocks if \n\n split fails
    if len(blocks) < 5:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        buffer = ''
        blocks = []
        for line in lines:
            buffer += ' ' + line
            if len(buffer.split()) >= 50:  # adjustable chunk size
                blocks.append(buffer.strip())
                buffer = ''
        if buffer:
            blocks.append(buffer.strip())

    # Final filtering
    return [{"clause": block} for block in blocks if len(block.split()) >= 10]

# --- Entry point ---

def extract_clauses_from_url(url):
    response = requests.get(url)
    file_bytes = response.content

    content_type = response.headers.get("Content-Type")
    mime_type, _ = mimetypes.guess_type(url)

    if not mime_type and content_type:
        mime_type = content_type

    # Detect and extract raw text
    if mime_type:
        if "pdf" in mime_type:
            raw_text = extract_text_from_pdf(file_bytes)
        elif "msword" in mime_type or "docx" in mime_type:
            raw_text = extract_text_from_docx(file_bytes)
        elif "plain" in mime_type or url.endswith(".txt"):
            raw_text = extract_text_from_txt(file_bytes)
        elif "message/rfc822" in mime_type or url.endswith(".eml"):
            raw_text = extract_text_from_eml(file_bytes)
        else:
            raw_text = extract_text_from_pdf(file_bytes)  # fallback
    else:
        raw_text = extract_text_from_pdf(file_bytes)  # fallback

    return split_into_clauses(raw_text)
