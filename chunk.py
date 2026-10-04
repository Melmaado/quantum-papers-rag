import re
from pathlib import Path
from pylatexenc.latex2text import LatexNodes2Text
import json

converter = LatexNodes2Text()
TEXT_DIR = Path("data/text")
CHUNKS_PATH = Path("data") / "chunks.json"

def split_sections(body):
    """Split a LaTeX body into (path, text) pairs, one per section.
    The first pair is ("Main text", ...) for the text before the first heading.
    Paths join nested headings with " > ", e.g. "Section > Subsection".
    """
    sections = []
    path = []
    pieces = re.split(r"\\((?:sub)*section)\*?\{([^}]+)\}", body)
    sections.append(("Main text", pieces[0]))
    for i in range(1, len(pieces), 3):
        title = " ".join(pieces[i+1].split())
        level = pieces[i].count("sub")
        path = path[:level] + [title]
        sections.append((" > ".join(path), pieces[i+2]))
    return sections

def latex_to_plain(tex):
    r"""Convert a LaTeX fragment to clean plain text.
    Each \href{url}{text} is replaced by its text before conversion,
    since pylatexenc can crash on it. After conversion with pylatexenc,
    image and citation placeholders are removed and whitespace is
    collapsed into single spaces.
    """
    tex = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", tex)
    plain = converter.latex_to_text(tex)
    plain = plain.replace("< g r a p h i c s >", "")
    plain = plain.replace("<cit.>", "")
    plain = " ".join(plain.split())
    return plain

def chunk_words(text, size = 150, overlap = 20):
    """Split text into chunks of `size` words, each overlapping the previous one by `overlap` words.
    Returns a list of strings. The last chunk may be shorter than `size`.
    """
    chunks = []
    step = size - overlap
    words = text.split()
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i+size]))
        if i + size >= len(words):
            break
    return chunks

assert chunk_words("a b c d e f g h i j", size=4, overlap=1) == ["a b c d", "d e f g", "g h i j"]
assert chunk_words("a b", size=4, overlap=1) == ["a b"]
assert chunk_words("a b c d e f g h i j k", size=4, overlap=1) == ["a b c d", "d e f g", "g h i j", "j k"]

all_chunks = []
for file in TEXT_DIR.glob("*.tex"):
    with open(file, encoding="utf-8") as f:
        text = f.read()
        body = text.split(r"\begin{document}",1)[1].split(r"\end{document}",1)[0]
        sections = split_sections(body)
        total_words = 0
        paper_chunks = 0
        for path, latex in sections:
            try:
                plain = latex_to_plain(latex)
                if plain:
                    total_words += len(plain.split())
                    section_chunks = chunk_words(plain)
                    for chunk in section_chunks:
                        all_chunks.append({"paper_id":file.stem, "section":path, "text":chunk})
                    paper_chunks+=len(section_chunks)
                else: 
                    continue
            except IndexError:
                print(f"It crashed because of an IndexError in {file.name}, {path}, let's go on")
        print(f"{file.stem}: {len(sections)} sections, {total_words} words, {paper_chunks} chunks")
with open(CHUNKS_PATH, "w", encoding="utf-8") as j:
    json.dump(all_chunks, j, ensure_ascii=False, indent=2)
print(f"Total: {len(all_chunks)}")