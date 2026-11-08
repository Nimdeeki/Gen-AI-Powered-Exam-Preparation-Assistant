"""Turn previous-year question papers (PYQs) into a clean list of questions.

Rule-based (heuristic) NLP with regular expressions: university papers follow a
small number of numbering styles (Q1, Q.1, 1., (a), ii) ...), so regex is fast,
free and accurate enough. Users can correct anything it misses in the app.
"""

import re
from dataclasses import dataclass

YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
# A non-question line that announces a paper, e.g. "End Semester Examination May 2023"
YEAR_HEADER_RE = re.compile(
    r"(?:exam|examination|paper|semester|session|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"\D{0,40}((?:19|20)\d{2})(?!\d)", re.I)

MAIN_Q_RE = re.compile(
    r"^(?:Q(?:ue(?:s(?:tion)?)?)?\s*\.?\s*(?:No\.?\s*)?(\d{1,2})[a-z]?|(\d{1,2}))\s*[\.\):\-]?\s+(.*)$",
    re.I)
# (a) / a) / ii) / a. Explain...  ("a." needs a space after it so "B.Tech" is not a sub-question)
SUB_Q_RE = re.compile(r"^\(?((?:[a-h])|(?:i{1,3}|iv|vi{0,3}|ix|x))\s*(?:\)\s*|\.\s+)(.*)$", re.I)
# A table cell holding only the marks of the question above: "10", "5 M", "10 CO3 L2"
MARKS_LINE_RE = re.compile(r"^(\d{1,2})\s*(?:M|marks?|(?:M\s*)?CO\s*\d+(?:\s*(?:BL|L|K)\s*\d)?)$", re.I)

INSTRUCTION_RE = re.compile(
    r"attempt|answer\s+(?:any|all|the\s+following)|all\s+questions|compulsory|max(?:imum)?\.?\s*marks"
    r"|time\s*(?:allowed)?\s*:|roll\s*no|candidates?|instructions?|figures\s+to\s+the\s+right"
    r"|assume\s+suitable|\bOR\b\s*$|^or$|^section\s+[a-z]\b|^part\s+[a-z]\b", re.I)

JUNK_LINE_RE = re.compile(
    r"^(page\s*\d+(\s*of\s*\d+)?|p\.?\s*t\.?\s*o\.?|-+\s*\d+\s*-+|\*+|end of paper"
    r"|CO\s*\d+|(?:BL|L|K)\s*\d|CO\s*\d+\s*(?:BL|L|K)\s*\d|q\.?\s*no\.?|questions?|marks?)$", re.I)
# Table-style papers put the question number alone on its own line: "1", "Q.2", "3)"
NUMBER_ONLY_RE = re.compile(r"^(?:Q(?:ue(?:s(?:tion)?)?)?\s*\.?\s*(?:No\.?\s*)?)?\d{1,2}[a-z]?\s*[\.\):]?$", re.I)

MARKS_PATTERNS = [
    re.compile(r"[\[\(]\s*(\d{1,2}(?:\.\d)?)\s*(?:marks?|m)?\s*[\]\)]\s*$", re.I),            # [5] (10 marks)
    re.compile(r"\b(\d{1,2})\s*marks?\b\.?\s*$", re.I),                                          # 5 marks
    re.compile(r"\s(\d{1,2})\s*(?:M|marks?)?\s*CO\s*\d+\s*(?:(?:BL|L|K)\s*\d)?\s*$", re.I),  # 5 CO2 L3
    re.compile(r"\s(\d{1,2})\s*M\s*$"),                                                        # 5M
]
TRAILING_CO_RE = re.compile(r"\s*\bCO\s*\d+\s*(?:(?:BL|L|K)\s*\d)?\s*$", re.I)

QUESTION_TYPES = [
    ("Compare / differentiate", r"\b(differentiate|distinguish|compare|contrast|difference|versus|vs\.?)\b"),
    ("Numerical / calculate", r"\b(calculate|compute|solve|numerical|evaluate|find the value)\b"),
    ("Code / implement", r"\b(code|program|implement|python|pseudo-?code)\b"),
    ("Diagram / architecture", r"\b(diagram|draw|architecture|flow\s*chart|sketch|illustrate)\b"),
    ("Short notes", r"\b(short notes?|write a note|notes? on)\b"),
    ("Definition / list", r"^(what is|what are|define|state|list|name|mention|enumerate)\b|\bdefine\b"),
    ("Explain / describe", r"\b(explain|describe|discuss|elaborate|justify|why|how|what)\b"),
]


@dataclass
class PYQuestion:
    text: str
    source: str
    year: int | None = None
    marks: float | None = None
    qtype: str = "Other"


def classify_question(text: str) -> str:
    low = text.lower()
    for label, pattern in QUESTION_TYPES:
        if re.search(pattern, low):
            return label
    return "Other"


def split_marks(text: str) -> tuple[str, float | None]:
    for pattern in MARKS_PATTERNS:
        m = pattern.search(text)
        if m:
            return text[:m.start()].strip(), float(m.group(1))
    return TRAILING_CO_RE.sub("", text).strip(), None


def detect_year(name: str, text: str) -> int | None:
    m = YEAR_RE.search(name) or YEAR_HEADER_RE.search(text[:2000]) or YEAR_RE.search(text[:1500])
    return int(m.group(1)) if m else None


def _blocks(text: str, start_year: int | None) -> list[tuple[str, str, int | None]]:
    """First pass: split lines into ('main' | 'sub', text, year) blocks."""
    blocks: list[list] = []
    year = start_year
    open_block = False          # may the next plain line continue the last question?
    pending_number = False      # saw a bare "1" / "Q.2" line; the next line is a question
    for raw in text.splitlines():
        line = raw.strip()
        if not line or JUNK_LINE_RE.match(line):
            continue
        marks_cell = MARKS_LINE_RE.match(line)
        if marks_cell and blocks:
            blocks[-1][1] += f" [{marks_cell.group(1)}]"
            open_block = False
            continue
        if NUMBER_ONLY_RE.match(line):
            pending_number = True
            continue
        main, sub = MAIN_Q_RE.match(line), SUB_Q_RE.match(line)
        if pending_number and not main and not sub:
            main = MAIN_Q_RE.match(f"0 {line}")
        pending_number = False
        if main:
            rest = main.group(3).strip()
            inner = SUB_Q_RE.match(rest)
            if inner:  # "Q1 (a) Explain ..." -> empty stem + sub-question
                blocks.append(["main", "", year])
                blocks.append(["sub", inner.group(2).strip(), year])
            else:
                blocks.append(["main", rest, year])
            open_block = True
        elif sub and len(line) > 3:
            blocks.append(["sub", sub.group(2).strip(), year])
            open_block = True
        else:
            header = YEAR_HEADER_RE.search(line)
            if header and len(line) < 100 and not line.endswith("?"):
                year = int(header.group(1))   # a new paper starts (combined PYQ PDFs)
                open_block = False
            elif INSTRUCTION_RE.search(line):
                open_block = False
            elif open_block and blocks:
                blocks[-1][1] = f"{blocks[-1][1]} {line}".strip()
    return [tuple(b) for b in blocks]


def extract_questions(name: str, text: str) -> list[PYQuestion]:
    year = detect_year(name, text)
    blocks = _blocks(text, year)
    questions: list[PYQuestion] = []
    stem = ""
    for i, (kind, body, q_year) in enumerate(blocks):
        if kind == "main":
            stem = body
            has_parts = i + 1 < len(blocks) and blocks[i + 1][0] == "sub"
            if has_parts:
                continue                 # the stem is only a heading for its parts
        elif len(body.split()) < 6 and stem and not INSTRUCTION_RE.search(stem):
            body = f"{stem.rstrip(':')}: {body}"   # "Write short notes on: Tokenization"
        body, marks = split_marks(re.sub(r"\s+", " ", body))
        if len(body.split()) < 3:
            continue
        if INSTRUCTION_RE.search(body) and len(body.split()) < 12:
            continue
        questions.append(PYQuestion(text=body, source=name, year=q_year, marks=marks,
                                    qtype=classify_question(body)))

    # The same question can appear twice (e.g. an "OR" choice repeated); keep the first
    seen, unique = set(), []
    for q in questions:
        key = (q.year, q.text.lower())
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique
