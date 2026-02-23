from app.search import search_titles
from app.lstm_predict import predict_next_words


def hybrid_autocomplete(query: str):
    matches = search_titles(query)
    prediction = predict_next_words(query)

    return {
        "query": query,
        "matches": matches,
        "prediction": prediction
    }