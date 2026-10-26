def is_hit(results, expected):
    """Return True if any retrieved chunk comes from one of the expected passages.
    There is a match between a chunk and a passage when both the paper_id and the section are equal
    """
    for chunk in results:
        for expectation in expected:
            if chunk['paper_id'] == expectation['paper_id'] and chunk['section'] == expectation['section']:
                return True
    return False


def is_paper_hit(results, expected):
    """Return True if any retrieved chunk comes from one of the expected papers.
    A chunk matches when its paper_id equals an expected paper_id, whatever its section.
    """
    for chunk in results:
        for expectation in expected:
            if chunk['paper_id'] == expectation['paper_id']:
                return True
    return False