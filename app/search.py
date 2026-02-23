TITLES_FILE = "autocomplete_lstm/data/titles.txt"

with open(TITLES_FILE, "r", encoding="utf-8") as f:
    TITLES = [line.strip().lower() for line in f if line.strip()]


def search_titles(query: str, top_k=5):
    query = query.lower().strip()

    results = []

    for title in TITLES:
        if title.startswith(query):
            results.append(title)

        if len(results) >= top_k:
            break

    return results