import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SOURCES_CONFIG = Path(__file__).parent.parent / "pdf_sources.json"

#change 1: experimented with chunk size

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

#for pattern matching of clause numbers in different formats 
_ROMAN = r"(?:(?:X{1,3}(?:IX|IV|V?I{0,3})|IX|IV|V?I{1,3}|VI{0,3}))"

SUB_SUBSECTION_PATTERN = re.compile(r"^(\d+\.\d+\.\d+)\.?\s+(.+)$")
SUBSECTION_PATTERN = re.compile(r"^(\d+\.\d+)\.?\s+(.+)$")
SECTION_PATTERN = re.compile(r"^(\d+)[\.\s]+(.+)$")
ROMAN_SECTION_PATTERN = re.compile(rf"^({_ROMAN})\.?\s+(.+)$", re.IGNORECASE)
MIXED_PATTERN = re.compile(rf"^(\d+)\.({_ROMAN})\.?\s+(.+)$", re.IGNORECASE) #some pdfs have a mix of numeric systems

#change 3: added more patterns to ignore such as headers footers and common irrelevant sections
#for cleaning up pages headers and footers 

_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")
# spaced-out capital headers 
_PAGE_HEADER_RE = re.compile(
    r"^\d*\s*[A-Z]\s+[A-Z]+(?:\s+[A-Z]\s*[A-Z]+)*\s*$"
)
# repeated document title headers that were breaking parsing
_DOC_TITLE_HEADERS = [
    re.compile(r"^\s*\d{4}\s+POLICY\s+ON\s+.+$", re.IGNORECASE),
    re.compile(r"^\s*POL\s+\d+\s+v[\d.]+.*$", re.IGNORECASE),
    re.compile(r"^\s*Global\s+Alliance\s+for\s+Genomics\s+(?:and|&)\s+Health.*$", re.IGNORECASE),
    re.compile(r"^\s*GA4GH\s+Data\s+Privacy\s+and\s+Security\s+Policy\s*$", re.IGNORECASE),
    re.compile(r"^\s*Framework\s+for\s+Responsible\s+Sharing\s+of\s+Genomic.*$", re.IGNORECASE),
    re.compile(r"^\s*FRAMEWORK\s+FOR\s+INVOLVING\s+AND\s+ENGAGING.*$", re.IGNORECASE),
    re.compile(r"^\s*Version\s+POL\s+\d+.*$", re.IGNORECASE),
]

#irrelevant sections common in many docs
IGNORE_TITLES = [
    "acknowledgements",
    "references",
    "contributors",
    "revision history",
    "resources",
    "appendix",
    "background",
    "purpose",
    "introduction",
    "overview",
    "context",
    "preamble",
    "conclusion",
    "discussion",
    "implementation mechanisms and amendments",
    "framework genesis and purpose",
]

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

def extract_pages(file_path: str) -> List[Tuple[int, str]]:
    #extract text and return page no and text
    reader = PdfReader(file_path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append((i, _clean_text(text)))
    return pages


def extract_pdf_text(file_path: str) -> str:
    #return cobined text 
    return "\n".join(text for _, text in extract_pages(file_path))



def _clean_title(title: str) -> str:
    #strip trailing dots, zero-width chars, leading/trailing whitespace
    return title.strip().strip(".")


def _match_heading(line: str) -> Optional[Dict[str, str]]:
    #matches against section numbering patterns
    m = SUB_SUBSECTION_PATTERN.match(line)
    if m:
        return {"id": m.group(1), "title": _clean_title(m.group(2)),
                "level": "subsubsection", "parent_prefix": m.group(1).rsplit(".", 1)[0]}

    m = MIXED_PATTERN.match(line)
    if m:
        cid = f"{m.group(1)}.{m.group(2)}"
        return {"id": cid, "title": _clean_title(m.group(3)),
                "level": "subsection", "parent_prefix": m.group(1)}

    m = SUBSECTION_PATTERN.match(line)
    if m:
        return {"id": m.group(1), "title": _clean_title(m.group(2)),
                "level": "subsection", "parent_prefix": m.group(1).split(".")[0]}

    m = SECTION_PATTERN.match(line)
    if m:
        return {"id": m.group(1), "title": _clean_title(m.group(2)),
                "level": "section", "parent_prefix": None}

    m = ROMAN_SECTION_PATTERN.match(line)
    if m:
        return {"id": m.group(1).upper(), "title": _clean_title(m.group(2)),
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
    return {
        "document_name": document_name,
        "chunk_id": chunk_id,
        "title": title,
        "content": content.strip(),
        "parent_id": parent_id,
        "level": level,
        "source_url": source,
        "page": page,
        "type": doc_type,
    }


def parse_clauses(
    text: str,
    source: str,
    doc_type: str = "policy",
    document_name: str = "",
) -> Tuple[List[Dict[str, Any]], str]:
    # parse text into clauses 

    #split entire pdf into individual lines 
    raw_lines = text.split("\n")
    lines: List[str] = []
    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            lines.append("")
            continue
        if lines and lines[-1] and not re.match(r"^[\dIVXivx]", stripped):
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
                result.append({**c, "chunk_id": f"{c['chunk_id']}_part{idx}", "content": part})
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
    doc_type: str = "policy",
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
    doc_type: str = "policy",
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
    clauses, unclaimed = parse_clauses(text, source, doc_type, document_name=doc_name)

    pages = extract_pages(file_path)
    _assign_pages(clauses, pages)

    clauses = _postprocess_clauses(clauses)
     
    #change 3: removed fallback text to avoid noise in retrieval
    """fallback = _fallback_chunks(unclaimed, source, doc_type=doc_type,
                                document_name=doc_name) if unclaimed else []"""
    fallback = []

    """if not clauses and not fallback:
        print(f"Warning: No clauses or text found in {source}.")
        fallback = _page_overlap_fallback(file_path, source, doc_type,
                                          document_name=doc_name)"""
    if not clauses:
        print(f"Warning: No clauses detected in {source}. Skipping document.")
        return []

    combined = clauses + fallback
    print(f"  {source}: {len(clauses)} clause chunk(s) + {len(fallback)} fallback chunk(s)")
    return combined


def _load_source_config() -> Dict[str, Dict[str, str]]:
    #loads metadata config 
    if SOURCES_CONFIG.exists():
        with open(SOURCES_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def fetch_all_pdfs(
    directory: str,
    doc_type: str = "policy",
) -> List[Dict[str, Any]]:
    #ingest all PDFs in a directory
    pdf_dir = Path(directory)
    if not pdf_dir.is_dir():
        print(f"Directory not found: {directory}")
        return []

    source_map = _load_source_config()
    all_chunks: List[Dict[str, Any]] = []
    #sorting for reproducibility 
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        print(f"Ingesting: {pdf_file.name}")
        cfg = source_map.get(pdf_file.name, {})
        all_chunks.extend(fetch_pdf_chunks(
            str(pdf_file), doc_type,
            source_url=cfg.get("source_url"),
            document_name=cfg.get("document_name"),
        ))

    print(f"Total chunks from PDFs: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":

    docs_dir = str(Path(__file__).parent.parent / "docs")
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None

    if pdf_path:
        chunks = fetch_pdf_chunks(pdf_path)
    else:
        chunks = fetch_all_pdfs(docs_dir)

    for c in chunks[:15]:
        print(f"[{c['chunk_id']}] {c['title']} "
              f"(level={c['level']}, parent={c['parent_id']}, page={c.get('page')})")
        print(f"  {c['content'][:200]}")
        print()
