import json
import sys
import re
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from ingestion.text_config import (
    SUB_SUBSECTION_PATTERN,
    SUBSECTION_PATTERN,
    SECTION_PATTERN,
    ROMAN_SECTION_PATTERN,
    MIXED_PATTERN,
    _PAGE_NUMBER_RE,
    _PAGE_HEADER_RE,
    _DOC_TITLE_HEADERS,
    IGNORE_TITLES,
    STOPWORDS,
    DOMAIN_TERMS,
)

SOURCES_CONFIG = Path(__file__).parent.parent / "pdf_sources.json"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
MAX_CLAUSE_CHARS = 1500

# CHUNK_SIZE = 400 
# CHUNK_OVERLAP = 80 
# MAX_CLAUSE_CHARS = 800 

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)

#removing appendix sections
def _ignore_fluff(title: str):
    t = title.lower()
    return any(ig in t for ig in IGNORE_TITLES)


def _is_header_or_footer(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _PAGE_NUMBER_RE.match(stripped):
        return True
    if _PAGE_HEADER_RE.match(stripped):
        return True
    return any(pattern.match(stripped) for pattern in _DOC_TITLE_HEADERS)

def _clean_text(text: str) -> str:
    #normalize bullet points, zero-width spaces, and excess whitespace
    text = text.replace("\u200b", "") 
    text = re.sub(r"[\u2022\u25cf\u25a0\u25aa\u25b8\u25ba]\s*", "", text) 
    text = re.sub(r"[^\S\n]+", " ", text) 
    text = re.sub(r"\n{3,}", "\n\n", text) 

    lines = text.split("\n")
    cleaned_lines = [line for line in lines if not _is_header_or_footer(line)]

    return "\n".join(cleaned_lines).strip()


def extract_keywords(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    domain_hits = [w for w in words if w in DOMAIN_TERMS]
    other = [w for w in words if w not in STOPWORDS and w not in set(domain_hits)]
    combined = domain_hits + other
    return list(dict.fromkeys(combined))[:12]

def extract_pages(file_path: str) -> List[Tuple[int, str]]:
    #extract text and return page no and text
    reader = PdfReader(file_path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append((i, _clean_text(text)))
    return pages

def extract_tables(pdf_path: str) -> List[Tuple[int, List[str]]]:
    #the consent toolkit docs contain tables 
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg_no, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            for table in tables:
                for row in table:
                    if row:
                        rows.append((pg_no,row))
    return rows

# skip header rows and metadata that pdfplumber picks up from non-content tables
_TABLE_SKIP_RE = re.compile(
    r"^(categories|consent\s+clauses?|consent\s+elements|summary\s+of\s+revisions"
    r"|date\s+effective|deliverable|version|special\s+thanks|policy\s+number)",
    re.IGNORECASE,
)

def table_rows_to_chunks(
    #making chunks out of extracted tables
    rows: List[Tuple[int, List[str]]],
    source: str,
    document_name: str,
    doc_type: str = "consent_toolkit",
) -> List[Dict[str, Any]]:
    chunks = []
    last_topic = "" 

    for i, (pg_no, row) in enumerate(rows):
        topic = (row[0] or "").strip().replace("\n", " ") if len(row) > 0 else ""
        clause = (row[1] or "").strip().replace("\n", " ") if len(row) > 1 else ""

        if not clause or len(clause) < 20:
            continue
        # skip header/metadata rows
        if _TABLE_SKIP_RE.match(topic) or _TABLE_SKIP_RE.match(clause):
            continue

        # carry forward topic from merged cells
        if topic:
            last_topic = topic
        else:
            topic = last_topic

        content = f"Topic: {topic}\nClause: {clause}"
        chunks.append(_make_chunk(
            chunk_id=f"table_{i}",
            title=topic,
            content=content,
            parent_id=None,
            level="consent_clause",
            source=source,
            page=pg_no,
            doc_type=doc_type,
            document_name=document_name,
        ))
    return chunks


def extract_pdf_text(file_path: str) -> str:
    #return combined text
    return "\n".join(text for _, text in extract_pages(file_path))


def _clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(
        r"\s+(?:\[)?(?:"
        r"C\s*ONSENT\s+P\s*OLICY"
        r"|D\s*ATA\s+P\s*RIVACY\s+AND\s+S\s*ECURITY\s+P\s*OLICY"
        r"|F\s*RAMEWORK\s+FOR\s+R\s*ESPONSIBLE\s+S\s*HARING"
        r")\]?\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    return title.strip().strip(".").strip("[]")

def _split_heading_title(title: str) -> Tuple[str, str]:
    #some PDFs inline the first sentence after the heading label
    parts = title.split(". ", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        #avoid splitting on abbreviations inside headings
        words = parts[0].split()
        if len(words) <= 8 and parts[1][:1].isupper():
            return parts[0].rstrip("."), parts[1].strip()
    words = title.split()
    if len(words) >= 6:
        starters = {"this", "the", "these", "it", "they", "in", "for", "with", "without"}
        connectors = {"of", "for", "and", "to", "in", "with", "without", "on", "at", "by"}
        for idx, word in enumerate(words):
            clean = word.strip(",;:").lower()
            if idx > 0 and clean in starters and word[:1].isupper():
                pre_words = words[:idx]
                if all(w.isupper() or w.istitle() or w.lower() in connectors for w in pre_words):
                    return " ".join(pre_words).rstrip("."), " ".join(words[idx:]).strip()
    return title, ""


def _match_heading(line: str) -> Optional[Dict[str, str]]:
    #matches against section numbering patterns
    m = SUB_SUBSECTION_PATTERN.match(line)
    if m:
        title, remainder = _split_heading_title(_clean_title(m.group(2)))
        return {"id": m.group(1), "title": title, "remainder": remainder,
                "level": "subsubsection", "parent_prefix": m.group(1).rsplit(".", 1)[0]}

    m = MIXED_PATTERN.match(line)
    if m:
        cid = f"{m.group(1)}.{m.group(2)}"
        title, remainder = _split_heading_title(_clean_title(m.group(3)))
        return {"id": cid, "title": title, "remainder": remainder,
                "level": "subsection", "parent_prefix": m.group(1)}

    m = SUBSECTION_PATTERN.match(line)
    if m:
        title, remainder = _split_heading_title(_clean_title(m.group(2)))
        return {"id": m.group(1), "title": title, "remainder": remainder,
                "level": "subsection", "parent_prefix": m.group(1).split(".")[0]}

    m = SECTION_PATTERN.match(line)
    if m:
        title, remainder = _split_heading_title(_clean_title(m.group(2)))
        return {"id": m.group(1), "title": title, "remainder": remainder,
                "level": "section", "parent_prefix": None}

    m = ROMAN_SECTION_PATTERN.match(line)
    if m:
        title, remainder = _split_heading_title(_clean_title(m.group(2)))
        return {"id": m.group(1).upper(), "title": title, "remainder": remainder,
                "level": "section", "parent_prefix": None}

    return None


def _make_chunk(
    chunk_id: str,
    title: str,
    content: str,
    parent_id: Optional[str],
    level: str,
    source: str,
    page: Optional[int] = None,
    doc_type: str = "policy",
    document_name: str = "",
) -> Dict[str, Any]:
    keywords = extract_keywords(f"{title} {content}")
    return {
        "document_name": document_name,
        "clause_id": chunk_id,
        "chunk_id": chunk_id,
        "title": title,
        "content": content.strip(),
        "parent_id": parent_id,
        "level": level,
        "source_url": source,
        "page": page,
        "type": doc_type,
        "keywords": keywords,
    }


def parse_clauses(
    text: str,
    source: str,
    doc_type: str = "policy",
    document_name: str = "",
) -> Tuple[List[Dict[str, Any]], str]:
    # parse text into clauses

    #some PDFs flatten headings into long lines insert newlines before
    text = re.sub(r"(?<!^)(\b[IVX]{1,4}\.\s+[A-Z])", r"\n\1", text)
    text = re.sub(r"(?<!^)(\b\d+\.\d+\.\d+\.\s+[A-Z])", r"\n\1", text)
    text = re.sub(r"(?<!^)(\b\d+\.\d+\.\s+[A-Z])", r"\n\1", text)
    text = re.sub(r"(?<!^)(\b\d+\.\s+[A-Z])", r"\n\1", text)

    #split entire pdf into individual lines 
    raw_lines = text.split("\n")
    lines: List[str] = []
    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            lines.append("")
            continue
        if lines and lines[-1] and not re.match(r"^(\d+\.|\(?\d+\)|[IVXivx]+\.)\s+", stripped):
            lines[-1] = lines[-1] + " " + stripped
        else:
            lines.append(stripped)


    chunks: List[Dict[str, Any]] = []
    unclaimed_parts: List[str] = []  #lines that didn't fit any section

    current_section: Optional[Dict[str, Any]] = None
    current_sub: Optional[Dict[str, Any]] = None
    section_ids: Dict[str, str] = {}
    ignore_mode = False

    def _flush_sub():
        nonlocal current_sub
        if current_sub and current_sub["content"].strip():
            current_sub["content"] = current_sub["content"].strip()
            chunks.append(current_sub)
        current_sub = None    #reset current subsection for next one

    def _flush_section():
        nonlocal current_section
        _flush_sub()
        if current_section and current_section["content"].strip():
            current_section["content"] = current_section["content"].strip()
            chunks.append(current_section)
        current_section = None


    for raw_line in lines:
        #mutiple spaces replaced with single space 
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        heading = _match_heading(line)
        if heading:
            level = heading["level"]
            cid = heading["id"]
            remainder = heading.get("remainder", "")

            #appendices are not helpful even as fallback text so raising a flag to ignore all of them
            if _ignore_fluff(heading["title"]):
                _flush_section()
                ignore_mode = True
                continue

            ignore_mode = False
            if level == "section":
                _flush_section()
                current_section = _make_chunk(
                    cid, heading["title"], "", None, "section", source,
                    doc_type=doc_type, document_name=document_name,
                )
                section_ids[cid] = cid
            elif level in ("subsection", "subsubsection") and current_section:
                _flush_sub()
                parent = heading["parent_prefix"]
                current_sub = _make_chunk(
                    cid, heading["title"], "",
                    parent if parent in section_ids else current_section["chunk_id"],
                    level, source, doc_type=doc_type, document_name=document_name,
                )
            else:
                _flush_section()
                current_section = _make_chunk(
                    cid, heading["title"], "", None, level, source,
                    doc_type=doc_type, document_name=document_name,
                )
                section_ids[cid] = cid
            if remainder:
                if current_sub is not None:
                    current_sub["content"] += " " + remainder
                elif current_section is not None:
                    current_section["content"] += " " + remainder
            continue

        if ignore_mode:
            continue
        if current_sub is not None:
            current_sub["content"] += " " + line
        elif current_section is not None:
            current_section["content"] += " " + line
        else:
            unclaimed_parts.append(line)

    _flush_section()
    return chunks, " ".join(unclaimed_parts).strip()


def _postprocess_clauses(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #split clauses longer than MAX_CLAUSE_CHARS
    result: List[Dict[str, Any]] = []
    for c in clauses:
        if len(c["content"]) > MAX_CLAUSE_CHARS:
            parts = splitter.split_text(c["content"])
            for idx, part in enumerate(parts, start=1):
                new_chunk = {**c, "chunk_id": f"{c['chunk_id']}_part{idx}", "content": part}
                new_chunk["keywords"] = extract_keywords(f"{c['title']} {part}")
                result.append(new_chunk)
        else:
            result.append(c)
    return result


def _fallback_chunks(
    #text that doesn't match any clause pattern 
    text: str,
    source: str,
    start_index: int = 1,
    doc_type: str = "policy",
    document_name: str = "",
) -> List[Dict[str, Any]]:
    #split unclaimed text into chunks
    parts = splitter.split_text(text)
    return [
        _make_chunk(
            chunk_id=f"fallback_{i}",
            title=f"Fallback chunk {i}",
            content=part,
            parent_id=None,
            level="fallback_chunk",
            source=source,
            doc_type=doc_type,
            document_name=document_name,
        )
        for i, part in enumerate(parts, start=start_index)
    ]


def _assign_pages(
    #assign pages numbers to clauses 
    clauses: List[Dict[str, Any]],
    pages: List[Tuple[int, str]],
) -> None:
    for c in clauses:
        if c.get("page") is not None:
            continue
        cid = c["chunk_id"].split("_part")[0]  #2_part3_part2 -> 2_part3
        for page_num, page_text in pages:
            if cid in page_text:
                c["page"] = page_num
                break


def _page_overlap_fallback(
    file_path: str,
    source: str,
    doc_type: str,
    document_name: str = "",
) -> List[Dict[str, Any]]:
    #last resort fallback split each page's text
    pages = extract_pages(file_path)
    chunks: List[Dict[str, Any]] = []
    idx = 1
    for page_num, page_text in pages:
        parts = splitter.split_text(page_text)
        for part in parts:
            chunks.append(_make_chunk(
                chunk_id=f"p{page_num}_c{idx}",
                title=f"Page {page_num} chunk {idx}",
                content=part,
                parent_id=None,
                level="fallback_chunk",
                source=source,
                page=page_num,
                doc_type=doc_type,
                document_name=document_name,
            ))
            idx += 1
    return chunks


def fetch_pdf_chunks(
    file_path: str,
    doc_type: str,
    source_url: Optional[str] = None,
    document_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    #Hybrid chunking: clause detection + splitting
    path = Path(file_path)
    if not path.exists():
        print(f"PDF not found: {file_path}")
        return []

    source = source_url or path.name
    doc_name = document_name or path.stem
    text = extract_pdf_text(file_path)
    clauses: List[Dict[str, Any]] = []
    if doc_type != "consent_toolkit":
        clauses, _ = parse_clauses(text, source, doc_type, document_name=doc_name)

    pages = extract_pages(file_path)
    _assign_pages(clauses, pages)

    clauses = _postprocess_clauses(clauses)

    # extract table-based consent clauses (consent toolkit docs only)
    table_chunks = []
    if doc_type == "consent_toolkit":
        table_rows = extract_tables(file_path)
        table_chunks = table_rows_to_chunks(table_rows, source, doc_name, doc_type) if table_rows else []

    fallback = []
    if not clauses and not table_chunks:
        print(f"Warning: No clauses detected in {source}. Skipping document.")
        return []

    combined = clauses + table_chunks + fallback
    print(f"  {source}: {len(clauses)} clause chunk(s) + {len(table_chunks)} table chunk(s) + {len(fallback)} fallback chunk(s)")
    return combined


def _load_source_config() -> Dict[str, Dict[str, str]]:
    #loads metadata config 
    if SOURCES_CONFIG.exists():
        with open(SOURCES_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def fetch_all_pdfs(
    directory: str,
) -> List[Dict[str, Any]]:
    #ingest all PDFs in a directory
    pdf_dir = Path(directory)
    if not pdf_dir.is_dir():
        print(f"Directory not found: {directory}")
        return []

    source_map = _load_source_config()
    all_chunks: List[Dict[str, Any]] = []

    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        #sorting for reproducibility 
        print(f"Ingesting: {pdf_file.name}")
        cfg = source_map.get(pdf_file.name, {})
        doc_type = cfg.get("doc_type","policy")
        all_chunks.extend(fetch_pdf_chunks(
            str(pdf_file), 
            doc_type,
            source_url=cfg.get("source_url"),
            document_name=cfg.get("document_name"),
        ))

    print(f"Total chunks from PDFs: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":

    docs_dir = str(Path(__file__).parent.parent / "docs")
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None

    if pdf_path:
        source_map = _load_source_config()
        cfg = source_map.get(Path(pdf_path).name, {})
        chunks = fetch_pdf_chunks(pdf_path, doc_type=cfg.get("doc_type", "policy"))
    else:
        chunks = fetch_all_pdfs(docs_dir)

    for c in chunks[:15]:
        print(f"[{c['chunk_id']}] {c['title']} "
              f"(level={c['level']}, parent={c['parent_id']}, page={c.get('page')})")
        print(f"  {c['content'][:200]}")
        print()
