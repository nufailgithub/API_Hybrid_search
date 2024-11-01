import os
import uuid
from fastapi import FastAPI, HTTPException
import json
from pathlib import Path
from PIL import Image
from typing import List, Dict, Union, Optional
from fastapi.middleware.cors import CORSMiddleware
import lancedb
from lancedb.pydantic import LanceModel, Vector
from torchvision.transforms.v2.functional import resize_image
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database initialization
uri = "data/sample-lancedb"
os.makedirs(uri, exist_ok=True)
db = lancedb.connect(uri)

# Load the Moondream2 model and tokenizer
model_id = "vikhyatk/moondream2"
revision = "2024-07-23"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Load the text_encoder model
text_model_id = "sentence-transformers/all-MiniLM-L6-v2"
text_encoder = SentenceTransformer(text_model_id)


# Define the schema for our products table
class Product(LanceModel):
    id: str
    title: str
    vector: Vector(384)  # Vector field for embeddings
    price: Optional[float] = None
    description: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    image_url: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None

def extract_image_vector(image_path: str):
    """Extract feature vector from an image using MoonDream."""
    try:
        img = Image.open(image_path).convert("RGB")
        enc_image = model.encode_image(img)
        return resize_image(enc_image, 384)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(384)  # Return zero vector on error

def extract_text_vector(text: str) -> np.ndarray:
    """Extract feature vector from text using Sentence Transformer"""
    try:
        return text_encoder.encode(text)
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return np.zeros(384)

def resize_vector(vector: np.ndarray, target_size: int) -> np.ndarray:
    """Resize vector to target dimension using interpolation"""
    # Ensure vector is 1D
    vector = vector.flatten()
    # Create indices for interpolation
    orig_indices = np.arange(len(vector))
    new_indices = np.linspace(0, len(vector) - 1, target_size)
    # Interpolate
    resized = np.interp(new_indices, orig_indices, vector)
    return resized

def load_products():
    """Load products from JSONL file."""
    products = []
    try:
        current_dir = Path(__file__).parent
        file_path = current_dir / "data/meta_Magazine_Subscriptions.jsonl"

        with open(file_path, "r") as file:
            for line in file:
                product = json.loads(line.strip())
                if 'id' not in product:
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


def combine_vectors(text_vector: np.ndarray, image_vector: np.ndarray,
                    text_weight: float = 0.5) -> np.ndarray:
    """Combine text and image vectors"""
    # Ensure both vectors are the same size
    assert len(text_vector) == len(image_vector) == 384

    # Combine vectors with weights
    combined = (text_weight * text_vector + (1 - text_weight) * image_vector)
    # Normalize the combined vector
    return combined / np.linalg.norm(combined)



def initialize_database():
    """Initialize the LanceDB table with products and their vectors."""
    try:
        products = load_products()
        table_name = "products_table"

        # Drop the existing table if it exists
        if table_name in db.table_names():
            db.drop_table(table_name)

        # Prepare data for insertion
        processed_data = []

        for product in products:
            # Extract and process text
            title = preprocess_description(product.get('title', ''))
            description = preprocess_description(product.get('description', ''))

            # Get text vector
            text_vector = extract_text_vector(title + " " + description)

            # Extract image vector (if exists)
            image_path = product.get('image_path')
            if image_path:
                image_vector = extract_image_vector(image_path)
            else:
                image_vector = np.zeros(384)  # Placeholder vector

            # Create structured data entry
            processed_item = {
                "id": product.get('id', str(uuid.uuid4())),
                "title": title,
                "vector": image_vector.tolist(),  # Convert numpy array to list
                "description": description,
                "price": product.get('price', ''),
                "category": product.get('category', ''),
                "brand": product.get('brand', ''),
                "image_url": product.get('image_url', ''),
                "rating": float(product.get('rating', 0.0)),
                "review_count": int(product.get('review_count', 0))
            }
            processed_data.append(processed_item)

        # Create the table with schema and data
        table = db.create_table(
            table_name,
            data=processed_data,
            mode="overwrite"
        )

        # Create full-text search index
        try:
            table.create_fts_index("title")
            table.create_fts_index("description")
        except Exception as e:
            print(f"Warning: Could not create FTS index: {str(e)}")

        return table

    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

# Initialize the database and get table reference
try:
    table = initialize_database()
except Exception as e:
    print(f"Failed to initialize database: {str(e)}")
    raise

@app.get("/")
async def root():
    try:
        results = table.to_pandas()
        products = []
        for _, row in results.iterrows():
            product = {
                'id': row['id'],
                'title': row['title'],
                'price': row['price'],
                'description': row['description'],
                'category': row['category'],
                'brand': row['brand'],
                'image_url': row['image_url'],
                'rating': row['rating'],
                'review_count': row['review_count']
            }
            products.append(product)

        return {
            "status": "success",
            "count": len(products),
            "products": products
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching products: {str(e)}"
        )


# @app.get("/search/{query}")
# async def search(query: str, limit: int = 5, text_weight: float = 0.5, similarity_threshold: float = 0.5):
#     try:
#         query_text_vector = extract_text_vector(query)
#         query_combined = combine_vectors(
#             query_text_vector,
#             np.zeros(384),  # Zero vector for image since query has no image
#             text_weight
#         )
#         print("Query combined", query_combined)
#
#         # Get the search results with distance scores (without selecting _distance)
#         base_query = table.search(query_combined).metric("cosine").limit(limit)
#
#         # Get distances from the base search
#         distances_df = base_query.to_pandas()
#
#         # Get full results with selected columns
#         results = base_query.select([
#             "title"
#         ]).to_pandas()
#
#         if not results.empty:
#             # Calculate similarity scores
#             similarities = [1 - float(d) for d in distances_df['_distance']]
#             results['similarity_score'] = similarities
#
#             # Filter based on similarity threshold
#             results = results[results['similarity_score'] >= similarity_threshold]
#
#         return {
#             "status": "success",
#             "count": len(results),
#             "results": results.to_dict('records')
#         }
#
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Search failed: {str(e)}"
#         )

@app.get("/search/{query}")
async def search(query: str, limit: int = 5, text_weight: float = 0.5, similarity_threshold: float = 0.5):
    try:
        # 1. Vector Extraction Verification
        query_text_vector = extract_text_vector(query)
        print("Query text vector shape:", query_text_vector.shape)
        print("Query text vector:", query_text_vector)

        # 2. Combined Vector Check
        query_combined = combine_vectors(
            query_text_vector,
            np.zeros(384),  # Zero vector for image since query has no image
            text_weight
        )
        print("Combined vector shape:", query_combined.shape)
        print("Combined vector:", query_combined)

        # 3. Search Execution with Detailed Logging
        base_query = table.search(query_combined).metric("cosine").limit(limit)

        # 4. Detailed Search Result Inspection
        print("Base query type:", type(base_query))

        # 5. Convert to Pandas with Error Handling
        try:
            distances_df = base_query.to_pandas()
            print("Distances DataFrame:")
            print(distances_df)
            print("Distances columns:", distances_df.columns)
        except Exception as pandas_error:
            print(f"Error converting to pandas: {pandas_error}")
            return {"status": "error", "message": "Failed to convert search results to DataFrame"}

        # 6. Results Extraction
        try:
            results = base_query.select(["title"]).to_pandas()
            print("Results DataFrame:")
            print(results)
            print("Results columns:", results.columns)
        except Exception as results_error:
            print(f"Error extracting results: {results_error}")
            return {"status": "error", "message": "Failed to extract search results"}

        # 7. Similarity Score Calculation with Robust Error Handling
        if not results.empty:
            try:
                # Check if '_distance' column exists
                if '_distance' not in distances_df.columns:
                    print("WARNING: '_distance' column not found in distances DataFrame")
                    return {"status": "error", "message": "Distance column missing"}

                similarities = [1 - float(d) for d in distances_df['_distance']]
                results['similarity_score'] = similarities

                # Filter based on similarity threshold
                results = results[results['similarity_score'] >= similarity_threshold]

                print(f"After filtering, {len(results)} results remain")
            except Exception as similarity_error:
                print(f"Error calculating similarities: {similarity_error}")
                return {"status": "error", "message": "Failed to calculate similarities"}

        # 8. Return Results
        return {
            "status": "success",
            "count": len(results),
            "results": results.to_dict('records') if not results.empty else []
        }

    except Exception as e:
        # 9. Comprehensive Error Logging
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
