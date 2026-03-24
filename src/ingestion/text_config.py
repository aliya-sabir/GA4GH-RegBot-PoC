import re

#for pattern matching of clause numbers in different formats
_ROMAN = r"(?:(?:X{1,3}(?:IX|IV|V?I{0,3})|IX|IV|V?I{1,3}|VI{0,3}))"

SUB_SUBSECTION_PATTERN = re.compile(r"^(\d+\.\d+\.\d+)\.?\s+(.+)$")
SUBSECTION_PATTERN = re.compile(r"^(\d+\.\d+)\.?\s+(.+)$")
SECTION_PATTERN = re.compile(r"^(\d+)[\.\s]+(.+)$")
ROMAN_SECTION_PATTERN = re.compile(rf"^({_ROMAN})\.\s+(.+)$", re.IGNORECASE)
MIXED_PATTERN = re.compile(
    rf"^(\d+)\.({_ROMAN})\.?\s+(.+)$", re.IGNORECASE
)

# for cleaning up pages headers and footers
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")
# spaced out capital headers
_PAGE_HEADER_RE = re.compile(
    r"^\d*\s*[A-Z]\s+[A-Z]+(?:\s+[A-Z]\s*[A-Z]+)*\s*$"
)
# repeated document title headers that were breaking parsing
_DOC_TITLE_HEADERS = [
    re.compile(
        r"^\s*Global\s+Alliance\s+for\s+Genomics\s+(?:and|&)\s+Health.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*GA4GH\s+Data\s+Privacy\s+and\s+Security\s+Policy\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*Framework\s+for\s+Responsible\s+Sharing\s+of\s+Genomic.*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*Clinical\s+Genomics\s*Consent\s+Clauses\s*$", re.IGNORECASE),
    re.compile(r"^\s*Version[:\s].*\d{4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*Approved[:\s].*\d{4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*D\d{3}\w?\s*/\s*v\s*[\d.]+.*$", re.IGNORECASE),
    re.compile(
        r"^\s*(?:Table:\s*)?Consent\s+Clauses\s+for\s+Large\s+Scale\s+Initiatives.*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*Nguyen\s+et\s+al\..*BMC\s+Medical\s+Ethics.*$", re.IGNORECASE),
    re.compile(r"^\s*https?://doi\.org/.*$", re.IGNORECASE),
    re.compile(
        r"^\s*R\s+E\s+S\s+E\s+A\s+R\s+C\s+H\s+A\s+R\s+T\s+I\s+C\s+L\s+E.*$",
        re.IGNORECASE,
    ),
]

#irrelevant sections common in many docs
IGNORE_TITLES = [
    "acknowledgements",
    "references",
    "contributors",
    "revision history",
    "appendix",
    "context",
    "preamble",
    "conclusion",
    "implementation mechanisms and amendments",
    "deliverable revision history",
    "additional file",
]

#change 5: added keywords as well for hybrid retrieval
STOPWORDS = {
    "the","and","or","a","an","to","of","for","in","on",
    "with","by","is","are","be","this","that","it","as","at",
    "from","not","will","can","may","shall","should","would",
    "has","have","had","been","was","were","its","their","they",
    "you","your","we","our","any","all","each","such","which",
    "who","what","when","where","how","than","but","about",
    "into","through","during","before","after","between","also",
    "only","very","just","there","here","other","more","most",
    "some","does","did","these","those","own","same","both",
    "being","could","might","nor","too","then","include",
    "including","use","used","using","make","made","given",
    "provide","provided","well","based","however","therefore",
    "need","case","way","part","able","apply","whether",
    "must","upon","within","without","take","set","per",
    "one","two","even","already","many","next","still",
}

DOMAIN_TERMS = {
    "withdrawal", "withdraw", "authorization", "informed", "participate",
    "sequencing", "genome", "variant", "variants", "genes",
    "anonymized", "pseudonymized", "identifiable", "coded", "linkage",
    "breach", "confidentiality", "identification",
    "oversight", "accountability", "governance", "regulatory", "lawful",
    "incidental", "disclosure", "findings", "diagnosis", "diagnostic",
    "safeguards", "datasets", "collection", "processing", "storage",
    "familial", "minors",
    "commercial", "discrimination", "recontact", "limitations",
    "dissemination", "proportionate", "registries",
}
