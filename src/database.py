import uuid
import dataclasses
import pathlib
import torch
from qdrant_client import QdrantClient, models

from config import (
    GENERATED_IMAGES_FOLDER,
    DATABASE_LOCATION,
    DATABASE_API_KEY,
    MODEL_EMBEDDING_SIZE,
)

def create_connection():
    print(f"Connecting to Qdrant ({DATABASE_LOCATION})")
    return QdrantClient(DATABASE_LOCATION, api_key=DATABASE_API_KEY, timeout=60)

def create_collection(qdrant: QdrantClient):
    return qdrant.create_collection(
        collection_name="kanji",
        vectors_config=models.VectorParams(
            size=MODEL_EMBEDDING_SIZE,
            distance=models.Distance.COSINE,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
        ),
    )

def index_collection(qdrant: QdrantClient):
    qdrant.update_collection(
        collection_name="kanji",
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )


def insert(qdrant: QdrantClient, font_name: str, kanji_dict: dict[str, torch.Tensor], standard_set: set[str]):
    return qdrant.upload_points(
        collection_name="kanji",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "kanji": kanji,
                    "is_standard": kanji in standard_set,
                    "font": font_name,
                },
            )
            for kanji, embedding in kanji_dict.items()
        ],
        batch_size=256,
    )


def search_vector(qdrant: QdrantClient, query_vector: torch.Tensor, limit: int=10):
    hits = qdrant.search(
        collection_name="kanji",
        # query_vector=query_vector,
        query_vector=query_vector.numpy(),
        limit=limit,
        with_payload=True,
    )
    return hits

@dataclasses.dataclass
class SearchResult:
    kanji: str
    font: str
    image_path: pathlib.Path
    score: float

def format_search_results(hits: list[models.ScoredPoint]) -> list[SearchResult]:
    formatted = []
    for point in hits:
        kanji, font = point.payload["kanji"], point.payload["font"]
        formatted.append(SearchResult(
            kanji = kanji,
            font = font,
            image_path = GENERATED_IMAGES_FOLDER / font / f"{kanji}.png",
            score = point.score,
        ))
    # assert sorted(formatted, key=lambda result: result.score, reverse=True) == formatted
    return formatted

if __name__ == "__main__":
    from qdrant_client import QdrantClient
    qdrant = QdrantClient(":memory:")

    assert create_collection(qdrant), "Failed to create collection"

    from config import ROOT
    from PIL import Image
    from generate_images import list_fonts, load_kanji_list, generate_images_for_font

    font_name, font = next(iter(list_fonts().items()))
    kanji_list = load_kanji_list()[:20]
    images = generate_images_for_font(font, kanji_list)

    from encoder import load_model, get_embeddings

    extractor, encoder = load_model()

    test_image = Image.open(ROOT / ".testing" / "drawing.png", "r")
    standard_set = set(kanji_list[:10])  # complete lie, but just for testing

    tensor = get_embeddings(extractor, encoder, list(images.values()) + [test_image])
    embeddings = {name: tensor[i] for i, name in enumerate(kanji_list)}
    drawing_embedding = tensor[-1]

    insert(qdrant, font_name, embeddings, standard_set)

    results = search_vector(qdrant, drawing_embedding)

    formatted = format_search_results(results)
    print(formatted)

    # Image.open(formatted[0].image_path).show()
    # breakpoint()
