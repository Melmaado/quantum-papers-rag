with open("papers.txt") as f:
    for line in f:
        paper_id = line.strip()
        if paper_id != "":
            print(f"https://arxiv.org/e-print/{paper_id}")
            break