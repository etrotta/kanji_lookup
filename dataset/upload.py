import os
import uuid
import pathlib
import polars as pl
from tqdm import tqdm
from qdrant_client import QdrantClient, models

STANDARD_KANJI_SET_FILE = pathlib.Path.cwd() / "kanji_joyo.txt"
EMBEDDINGS_PARQUET_FILE = pathlib.Path.cwd() / "dataset/kanji_embeddings.parquet"

DATABASE_LOCATION = os.getenv("QDRANT_URL", 'localhost')
DATABASE_API_KEY = os.getenv("QDRANT_API_KEY")

MODEL_EMBEDDING_SIZE = 768
BATCH_SIZE = 256

def verify_paths():
    if not STANDARD_KANJI_SET_FILE.is_file():
        print("Standard Kanji list File not found, searched at:", str(STANDARD_KANJI_SET_FILE.resolve()))
        raise Exception("Missing required file, please download it and update the path to point to it.")

    if not EMBEDDINGS_PARQUET_FILE.is_file():
        print("Embeddings file not found, searched at:", str(EMBEDDINGS_PARQUET_FILE.resolve()))
        raise Exception("Missing required file, please download it and update the path to point to it.")


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

# Create with indexing disabled, then set it to index after we are done inserting records

def index_collection(qdrant: QdrantClient):
    qdrant.update_collection(
        collection_name="kanji",
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )


def insert(qdrant: QdrantClient, df: pl.DataFrame, standard_set: set[str]):
    payload = pl.struct(
        kanji = pl.col("kanji"),
        font = pl.col("font"),
        is_standard = pl.col("kanji").is_in(standard_set),
    ).alias("payload")

    return qdrant.upsert(
        collection_name="kanji",
        points=models.Batch(
            ids=[str(uuid.uuid4()) for _ in range(BATCH_SIZE)],
            vectors=df.get_column("embedding").to_numpy(),
            payloads=df.select(payload).get_column("payload").to_list(),
        )
    )

def get_standard_kanji_set() -> set[str]:
    file = STANDARD_KANJI_SET_FILE
    return set(file.read_text(encoding="UTF-8").splitlines())

def upload_embeddings():
    qdrant = create_connection()
    assert create_collection(qdrant), "Failed to create collection"

    standard_set = get_standard_kanji_set()
    df = pl.read_parquet(EMBEDDINGS_PARQUET_FILE)
    
    print(
        f"Uploading a total of {len(df)} embeddings over {len(df) // BATCH_SIZE} batches of "
        f"{BATCH_SIZE} for the following fonts: {df.get_column('font').unique(maintain_order=True).to_list()}"
    )

    _checkpoint = 0
    for df_slice in tqdm(df.iter_slices(BATCH_SIZE)):
        _checkpoint += 1
        if _checkpoint > 0:
            insert(qdrant, df_slice, standard_set)
        # if _checkpoint > 50:
        #     break
    
    index_collection(qdrant)


if __name__ == "__main__":
    # DATABASE_LOCATION = ":memory:"
    # DATABASE_API_KEY = None
    # STANDARD_KANJI_SET_FILE = pathlib.Path.cwd() / "data/inputs/text/kanji_joyo.txt"
    # EMBEDDINGS_PARQUET_FILE = pathlib.Path.cwd() / "dataset/kanji_embeddings.parquet"

    verify_paths()

    upload_embeddings()