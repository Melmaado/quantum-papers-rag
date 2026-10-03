import re
from pathlib import Path

TEXT_DIR = Path("data/text")

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

for file in TEXT_DIR.glob("*.tex"):
    with open(file, encoding="utf-8") as f:
        text = f.read()
        body = text.split(r"\begin{document}",1)[1].split(r"\end{document}",1)[0]
        print(file.name, len(split_sections(body)))