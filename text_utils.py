import re


def tokenize(text):
    """Lowercase the text and split it into words, dropping punctuation."""
    text = text.lower()
    text = re.findall(r"\w+",text)
    return text