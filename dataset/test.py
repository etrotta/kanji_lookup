from pathlib import Path
import torch
from torch import cosine_similarity
import polars as pl
from PIL import Image

import sys
# sys.path.append(str((Path.cwd() / 'src').resolve()))
sys.path.append(str((Path(__file__) / '../src').resolve()))
from encoder import load_model, get_embeddings  # noqa

lazy_df = pl.scan_parquet("dataset/kanji_embeddings.parquet")

data = lazy_df.group_by("font").head().collect()

font_labels = data.get_column("font")
kanji_labels = data.get_column("kanji")
tensors = torch.as_tensor(data.get_column("embedding"))

# test_image_files = list((Path.cwd() / "dataset").glob("test_*.png"))
test_image_files = list(Path(__file__).parent.glob("test_*.png"))

_test_labels = [file.stem.split("_")[1] for file in test_image_files]
test_images = [Image.open(file, 'r') for file in test_image_files]

extractor, model = load_model()
test_embeddings = get_embeddings(extractor, model, test_images)

def transform(tensor):
    return tensor * 0.5 + 0.5

def score(a, b):
    return cosine_similarity(transform(a), transform(b), dim=0)

guesses = [
    (label, score(embeddings, reference_embedding), kanji, font)
    for label, embeddings in zip(_test_labels, test_embeddings)
    for font, kanji, reference_embedding in zip(font_labels, kanji_labels, tensors)
]

guesses.sort(key=lambda t: (t[0], -t[1]))

print(*guesses[:10], sep='\n')
half = len(guesses) // 2
print(*guesses[half:half+10], sep='\n')

