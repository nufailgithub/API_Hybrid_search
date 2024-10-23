import uuid
from fastapi import FastAPI, HTTPException
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Union
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a pre-trained ResNet model for image feature extraction
resnet = models.resnet50(weights='IMAGENET1K_V1')
resnet.eval()  # Set the model to evaluation mode

# Image preprocessing (resize, normalize for ResNet)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_image_vector(image_path: str):
    """Extract feature vector from an image using ResNet."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        features = resnet(img_tensor)

    return features.squeeze().numpy()  # Convert to NumPy array


def load_products():
    """Load products from JSONL file."""
    products = []
    try:
        current_dir = Path(__file__).parent
        file_path = current_dir / "data/meta_Magazine_Subscriptions.jsonl"

        with open(file_path, "r") as file:
            for line in file:
                # products.append(json.loads(line.strip()))
                #remove if data had own id when retrieving from the db
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


def combine_vectors(text_vector, image_vector, text_weight=0.5, image_weight=0.5):
    """Combine text and image vectors into a single hybrid vector."""
    return text_weight * text_vector + image_weight * image_vector


# Load products once at startup
products = load_products()

# Prepare descriptions for TF-IDF and images for ResNet
product_descriptions = []
image_vectors = []

for product in products:
    # Process descriptions
    desc = product.get('title', '')
    processed_desc = preprocess_description(desc)
    product_descriptions.append(processed_desc)

    # Extract image vector (if image path exists)
    image_path = product.get('image_path', None)  # Assuming products have an 'image_path'
    if image_path:
        image_vector = extract_image_vector(image_path)
    else:
        image_vector = np.zeros((1000,))  # Placeholder vector if no image exists
    image_vectors.append(image_vector)

# Create and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    strip_accents='unicode',
    lowercase=True,
    stop_words='english',
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95  # Ignore terms that appear in more than 95% of documents
)
text_tfidf_matrix = vectorizer.fit_transform(product_descriptions)

# Apply PCA to reduce the text vector dimensions to 1000 to match the image vectors
pca = PCA(n_components=1000)
text_vectors_reduced = pca.fit_transform(text_tfidf_matrix.toarray())

# Combine text and image vectors into hybrid vectors
hybrid_vectors = [
    combine_vectors(text_vector, image_vector)
    for text_vector, image_vector in zip(text_vectors_reduced, image_vectors)
]


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
            if query.lower() in preprocess_description(p.get('title', '')).lower()
        ]
        print("KEYWORD SEARCH :", keyword_results)

        # Text-based semantic search using TF-IDF
        query_text_vector = vectorizer.transform([query]).toarray()
        query_text_vector_reduced = pca.transform(query_text_vector)

        query_image_vector = np.zeros((1000,))  # Assuming no image is being queried
        query_hybrid_vector = combine_vectors(query_text_vector_reduced[0], query_image_vector)

        # Calculate cosine similarity between query vector and hybrid product vectors
        cosine_similarities = cosine_similarity([query_hybrid_vector], hybrid_vectors).flatten()

        # Get top results based on cosine similarity
        similar_indices = cosine_similarities.argsort()[:-limit - 1:-1]
        semantic_results = [products[i] for i in similar_indices]

        print("SEMANTIC RESULTS :", semantic_results)

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
