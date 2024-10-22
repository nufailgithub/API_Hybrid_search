
import uuid

from fastapi import FastAPI, HTTPException
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from typing import List, Dict, Union

app = FastAPI()


def load_products():
    """Load products from JSONL file."""
    products = []
    try:
        current_dir = Path(__file__).parent
        print("CURRENT-PATH", current_dir)
        file_path = current_dir / "data/meta_Magazine_Subscriptions.jsonl"

        with open(file_path, "r") as file:
            for line in file:
                product = json.loads(line.strip())
                product['id'] = str(uuid.uuid4())
                products.append(product)
        return products
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Products data file not found at {file_path}"
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing products data")


def preprocess_description(description: Union[str, List, None]) -> str:
    """Convert description to string format, handling various input types."""
    if description is None:
        return ""
    if isinstance(description, list):
        return " ".join(str(item) for item in description)
    return str(description)


# Load products once at startup
products = load_products()

# Prepare descriptions for TF-IDF
product_descriptions = []
for product in products:
    desc = product.get('description', '')
    processed_desc = preprocess_description(desc)
    product_descriptions.append(processed_desc)

# Create and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    strip_accents='unicode',
    lowercase=True,
    stop_words='english',
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95  # Ignore terms that appear in more than 95% of documents
)
tfidf_matrix = vectorizer.fit_transform(product_descriptions)


@app.get("/")
async def root():
    return {
        "status": "success",
        "count": len(products),
        "products": products
    }

@app.get("/search/{query}")
async def search(query: str, limit: int = 5):
    try:
        # Keyword search (case insensitive)
        keyword_results = [
            p for p in products
            if query.lower() in preprocess_description(p.get('description', '')).lower()
        ]
        print("KEYWORD SEARCH :", keyword_results)

        # Semantic search
        query_vector = vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort()[:-limit - 1:-1]
        semantic_results = [products[i] for i in similar_indices]

        print("SEMENTIC RESULTS :", semantic_results)



        # Combine and deduplicate results
        combined_results = list({p['id']: p for p in (keyword_results + semantic_results)}.values())

        print("COMBINED RESULTS :", combined_results)

        return {
            "status": "success",
            "count": len(combined_results),
            "results": combined_results[:limit]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)



