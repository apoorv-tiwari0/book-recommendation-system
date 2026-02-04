from fastapi import FastAPI
from app.recommender import BookRecommender
from app.schemas import QueryRequest, RecommendationResponse

app = FastAPI(
    title="Book Recommendation API",
    version="1.0"
)

recommender = BookRecommender()


@app.post("/recommend", response_model=RecommendationResponse)
def recommend_books(request: QueryRequest):
    results = recommender.recommend(
        request.query,
        request.top_k
    )
    return {"results": results}
