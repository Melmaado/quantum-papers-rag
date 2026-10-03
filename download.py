from pathlib import Path
import requests
import time

RAW_DIR = Path("data")/"raw"
Path(RAW_DIR).mkdir(parents=True,exist_ok=True)
with open("papers.txt") as f:
    for line in f:
        paper_id = line.strip()
        if paper_id != "":
            url = f"https://arxiv.org/e-print/{paper_id}"
            out_path = RAW_DIR / f"{paper_id}.tar.gz"
            if not out_path.exists():
                response = requests.get(url)
                time.sleep(3)
                print(response.status_code)
                if response.status_code==200:
                    with open(out_path,"wb") as paper:
                        paper.write(response.content)