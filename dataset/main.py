"""Convert torch embeddings .pt files into a .parquet dataset

Note: I am aware that it currently does not works with HuggingFace `datasets`.
I don't care, and you can load with just about any Arrow-compliant library like pola.rs or even pandas.
See: https://github.com/huggingface/datasets/issues/5706
"""


from pathlib import Path

import torch
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
# import datasets

# Polars columns:
    # - font : pl.Enum
    # - character : pl.String
    # - embedding : pl.List(pl.Float32)

# Arrow Columns:
    # "font": pa.dictionary(pa.uint8(), pa.string()),
    # "kanji": pa.string(),
    # "embedding": pa.list_(pa.float32(), EMBEDDING_SIZE),


# the `embedding` should be a pl.Array(pl.Float32, EMBEDDING_SIZE) but they do not support writing it to files yet 
# I kinda wish I could use Utf16 instead of Utf8 for the `kanji` column, but it is not supported by arrow nor polars

EMBEDDING_SIZE = 768

EMBEDDINGS_FOLDER = Path.cwd() / "data" / "generated" / "embeddings"


# Data format:
# sub-folders with the name of each font
# # batch_{i}.pt => pytorch stacked tensor containing a batch of embeddings, represented as a tensor of shape (n, EMBEDDING_SIZE)
# # batch_{i}.txt => labels for these tensors
# E.g.
# embeddings/KaiseiTokumin-Regular/batch_0.pt -> (3, EMBEDDING_SIZE) tensor
# embeddings/KaiseiTokumin-Regular/batch_0.txt -> "猫\n狐\n狼"
data = []
for font_folder in EMBEDDINGS_FOLDER.iterdir():
    for tensor_file in font_folder.glob("*.pt"):
        tensor = torch.load(tensor_file, weights_only=True)
        labels = tensor_file.with_suffix(".txt").read_text("UTF-8").splitlines()
        assert len(tensor) == len(labels)
        data.append((font_folder, labels, tensor))

kanji = pl.concat([pl.Series("kanji", batch[1]) for batch in data])

# EmbeddingArray = pl.Array(pl.Float32, EMBEDDING_SIZE)
# tensors = pl.concat([pl.Series("embedding", batch[2].numpy(), dtype=EmbeddingArray) for batch in data])
# pyo3_runtime.PanicException: not yet implemented: Writing FixedSizeList to parquet not yet implemented
EmbeddingList = pl.List(pl.Float32)
tensors = pl.concat([pl.Series("embedding", batch[2].numpy(), dtype=EmbeddingList) for batch in data])

FontEnum = pl.Enum(folder.name for folder in EMBEDDINGS_FOLDER.iterdir())
fonts = pl.concat([pl.repeat(batch[0].name, len(batch[1]), dtype=FontEnum, eager=True) for batch in data])

df = pl.DataFrame(
    {
        "font": fonts,
        "kanji": kanji,
        "embedding": tensors,
    }
)

del data

DATASET_PATH = Path.cwd() / "dataset" / "kanji_embeddings.parquet"

# df.write_parquet(DATASET_PATH, statistics=True)
table = df.to_arrow()
del df
table = table.cast(pa.schema({
    "font": pa.dictionary(pa.uint8(), pa.string()),
    "kanji": pa.string(),
    "embedding": pa.list_(pa.float32(), EMBEDDING_SIZE),
}))

# compressions = [
#     'snappy',  # hard to tell the speed since it was the first but 197MB
#     'gzip',  # rather fast, 159MB
#     'brotli',  # that one took ages holy shit, 158MB though
#     'zstd',  # kinda slow, 158MB
#     'lz4',  # descent, 173MB
#     'none',  # fast but 227 MB
# ]
# for compression in compressions:
#     pq.write_table(table, DATASET_PATH.with_stem(DATASET_PATH.stem + compression), compression=compression)

# Note: The font and kanji take < 1 MB, the bulk of the dataset size is in the embeddings
# To my surprise, it looks like some algorithms are able to compress the embeddings by quite a lot (up to ~30% !)

pq.write_table(table, DATASET_PATH, compression="gzip")

# dataset = datasets.Dataset(table)
# I refuse to believe that they do not support Enums
del table
# dataset.push_to_hub(...)

# test = pl.read_parquet(DATASET_PATH)