# TODO Move the imports to the functions that need them to avoid importing unnecessary things?

import typing
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import torch

from config import (
    IMAGES_FOLDER,
    EMBEDDINGS_FOLDER,
)

from generate_images import (
    load_kanji_list,
    list_fonts,
    generate_images_for_font,
)
from encoder import (
    load_model,
    get_embeddings,
)
from database import (
    create_connection,
    create_collection,
    insert,
    search,
    format_search_results,
)

T = typing.TypeVar("T")


def batched(original: list[T], group_size: int) -> list[list[T]]:
    groups = []
    for i in range(0, len(original), group_size):
        groups.append(original[i : i + group_size])
    return groups


def generate_images():
    fonts = list_fonts()
    kanji_list = load_kanji_list()[:200]
    kanji_batches = batched(kanji_list, 32)

    # for font_name, font in tqdm(fonts.items()):
    for font_name in tqdm(fonts):
        font = fonts[font_name]
        out_folder = IMAGES_FOLDER / font_name
        out_folder.mkdir(exist_ok=True, parents=True)

        for kanji_batch in tqdm(kanji_batches):
            images = generate_images_for_font(font, kanji_batch)
            for kanji, img in images.items():
                img.save(out_folder / f"{kanji}.png")


def generate_embeddings():
    extractor, encoder = load_model()
    image_folders = list(IMAGES_FOLDER.iterdir())

    for folder in tqdm(image_folders):
        font_embeddings_folder = EMBEDDINGS_FOLDER / folder.name
        font_embeddings_folder.mkdir(exist_ok=True, parents=True)
        image_files = folder.glob("*.png")
        batches = batched(list(image_files), 32)
        for batch in tqdm(batches):
            _labels = [file.stem for file in batch]
            images = [Image.open(file, "r") for file in batch]

            tensor = get_embeddings(extractor, encoder, images)
            for kanji, embedding in zip(_labels, tensor):
                out_file = font_embeddings_folder / f"{kanji}.pt"
                torch.save(embedding, out_file)


def populate_database():
    qdrant = create_connection()
    assert create_collection(qdrant), "Failed to create collection"

    embedding_folders = list(EMBEDDINGS_FOLDER.iterdir())

    for folder in tqdm(embedding_folders):
        embeddings = list(folder.glob("*.pt"))
        for batch in tqdm(batched(embeddings, 64)):
            embeddings = {file.stem: torch.load(file, weights_only=True) for file in batch}

            insert(qdrant, folder.name, embeddings)


def search_file(drawing_file: Path):
    qdrant = create_connection()
    extractor, encoder = load_model()

    image = Image.open(drawing_file, "r").convert("L")
    tensor = get_embeddings(extractor, encoder, [image])
    results = search(qdrant, tensor[0], limit=20)

    formatted = format_search_results(results)
    print(f"Search Results for {drawing_file.stem}:")
    print([(result.font, result.kanji) for result in formatted])


def search_folder(folder: Path):
    qdrant = create_connection()
    extractor, encoder = load_model()

    labels, images = [], []
    for file in folder.glob("*.png"):
        labels.append(file.stem)
        image = Image.open(file, "r").convert("L")
        images.append(image)

    tensor = get_embeddings(extractor, encoder, images)

    for label, embedding in zip(labels, tensor):
        results = search(qdrant, embedding, limit=20)

        formatted = format_search_results(results)
        print(f"Search Results for {label}:")
        print([(result.font, result.kanji) for result in formatted])


if __name__ == "__main__":
    from cli import parser
    args = parser.parse_args()
    def _search(path: Path):
        if path.is_file():
            search_file(path)
        elif path.is_dir():
            search_folder(path)
        else:
            raise Exception(f'Could not find a file nor a folder at Path "{path.resolve()}"')
    functions = {
        "generate_images": generate_images,
        "generate_embeddings": generate_embeddings,
        "populate_database": populate_database,
        "search": lambda : _search(args.input),
    }
    functions[args._name]()
