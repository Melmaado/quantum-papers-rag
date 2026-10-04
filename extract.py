import tarfile
from pathlib import Path
import re

def remove_comment_lines(text):
    """Return the text without its full-line LaTeX comments (lines starting with %).
    Inline comments at the end of a line are kept.
    """
    lines = []
    for line in text.splitlines():
        if not line.lstrip().startswith("%"):
            lines.append(line)
    return "\n".join(lines)

def flatten(text,tar):
    r"""Return the text with each \input or \include replaced by the included file.
    Included files are read from the tar archive, and their comment lines are removed.
    They are themselves flattened.
    Inclusions of files missing from the archive are dropped.
    """
    document = []
    for line in text.splitlines(): 
        match = re.search(r"\\(?:input|include)\{([^}]+)\}",line)
        if match:
            file_name = match.group(1)
            if not file_name.endswith(".tex"):
                file_name = match.group(1)+".tex"
            if file_name in tar.getnames():
                raw_text = tar.extractfile(file_name).read().decode("utf-8", errors="ignore")
                content = remove_comment_lines(raw_text)
                content = flatten(content, tar)
                document.append(content)
        else:
            document.append(line)
    return "\n".join(document)


RAW_DIR = Path("data") / "raw"
TEXT_DIR = Path("data") / "text"
Path(TEXT_DIR).mkdir(parents=True, exist_ok = True)

for archive in RAW_DIR.glob("*.tar.gz"):
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            for name in tar.getnames():
                if name.endswith(".tex"):
                    content = tar.extractfile(name).read().decode("utf-8", errors="ignore")
                    if r"\begin{document}" in content:
                        clean_content = remove_comment_lines(content)
                        flat_content = flatten(clean_content, tar)
                        print(archive, name, "->", f"{len(flat_content)} characters")
                        paper_id = archive.name.removesuffix(".tar.gz")
                        out_path = TEXT_DIR / f"{paper_id}.tex"
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(flat_content)
                        break

    else:
        print(f"{archive.name} not an archive")