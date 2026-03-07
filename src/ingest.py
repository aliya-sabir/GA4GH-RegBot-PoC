import re
import requests
from bs4 import BeautifulSoup

GA4GH_FRAMEWORK_URL = "https://www.ga4gh.org/framework/"


def fetch_chunks(URL:str = GA4GH_FRAMEWORK_URL) -> list[dict]:
    #Fetch document and exract all clauses including subclauses

    response = requests.get(URL, timeout=15)
    if response.status_code != 200:
        print(f"Failed to fetch document: {response.status_code}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    content_area = soup.find("div",class_="entry-content") or soup.find("body")

    chunks=[]
    current_section=None
    current_subsections=[]

    for el in content_area.find_all(['h2', 'h3','p','ul','ol']):
        text=re.sub(r"\s+", " ", el.get_text(separator=" ")).strip()

        if not text:
            continue

        if el.name == "h2":
            if current_section:
                chunks.append(current_section)
                chunks.extend(current_subsections)
                current_subsections = []
            m = re.match(r"^(\d+)\s+(.+)$",text)
            if m:
                current_section = {
                    "chunk_id": m.group(1),
                    "title": text,
                    "content": "",
                    "parent_id": None,
                    "level": "section",
                    "source_url": URL,
                }
            else:
                current_section=None

        elif el.name == "h3":
            m = re.match(r"^(\d+\.\d+)\.?\s+(.+)$",text)
            if m and current_section:
                current_subsections.append({
                    "chunk_id": m.group(1),
                    "title":text,
                    "content": "",
                    "parent_id": current_section["chunk_id"],
                    "level": "subsection",
                    "source_url": URL,
                })
        elif el.name in ["p", "ul", "ol"]:
            if current_subsections:
                current_subsections[-1]["content"] += " " + text
            elif current_section:
                current_section["content"] += " " + text
                
    if current_section:
        chunks.append(current_section)
        chunks.extend(current_subsections)
    
    for c in chunks:
        c['content'] = c['content'].strip()
    
    return chunks


if __name__ == "__main__":
    chunks = fetch_chunks()
    for c in chunks[:3]:
        print(f"[{c['chunk_id']}] {c['title']} (level={c['level']}, parent={c['parent_id']})")
        print(f"  {c['content'][:120]}...")
        print()
   