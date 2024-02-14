# TODO Move the imports to the functions that need them to avoid importing unnecessary things?

import typing
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import torch

from config import (
    GENERATED_IMAGES_FOLDER,
    GENERATED_EMBEDDINGS_FOLDER,
    MODEL_EMBEDDING_SIZE,
    CALIBRATION_IMAGES_FOLDER,
    CALIBRATION_FILE,
)

from generate_images import (
    get_standard_kanji_set,
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
    search_vector,
    format_search_results,
)

T = typing.TypeVar("T")

GENERATE_IMAGES_BATCH_SIZE = 64
GENERATE_EMBEDDINGS_BATCH_SIZE = 256

def batched(original: list[T], group_size: int) -> list[list[T]]:
    groups = []
    for i in range(0, len(original), group_size):
        groups.append(original[i : i + group_size])
    # return [groups[0]]  # Return only the first group of each batch for testing
    return groups


def generate_images():
    fonts = list_fonts()
    print(f"Using the following fonts: {fonts.keys()}")

    kanji_list = load_kanji_list()
    print(f"Processing a total of {len(kanji_list)} Kanji in batches of {GENERATE_IMAGES_BATCH_SIZE}")
    kanji_batches = batched(kanji_list, GENERATE_IMAGES_BATCH_SIZE)

    for font_name in tqdm(fonts):
        font = fonts[font_name]
        out_folder = GENERATED_IMAGES_FOLDER / font_name
        out_folder.mkdir(exist_ok=True, parents=True)

        for kanji_batch in tqdm(kanji_batches):
            images = generate_images_for_font(font, kanji_batch)
            for kanji, img in images.items():
                img.save(out_folder / f"{kanji}.png")


def generate_embeddings():
    extractor, encoder = load_model()
    image_folders = list(GENERATED_IMAGES_FOLDER.iterdir())
    print(f"Generating embeddings for the following fonts: {tuple(folder.name for folder in image_folders)}")

    for folder in tqdm(image_folders):
        font_embeddings_folder = GENERATED_EMBEDDINGS_FOLDER / folder.name
        font_embeddings_folder.mkdir(exist_ok=True, parents=True)
        image_files = list(folder.glob("*.png"))
        print(f"Found {len(image_files)} images for font {folder.name}")

        batches = batched(list(image_files), GENERATE_EMBEDDINGS_BATCH_SIZE)
        for i, batch in enumerate(tqdm(batches)):
            _labels = [file.stem for file in batch]
            images = [Image.open(file, "r") for file in batch]
            tensor_out_file = font_embeddings_folder / f"batch_{i}.pt"
            labels_out_file = font_embeddings_folder / f"batch_{i}.txt"

            tensor = get_embeddings(extractor, encoder, images)

            labels_out_file.write_text("\n".join(_labels), encoding="UTF-8")
            torch.save(tensor, tensor_out_file)


def create_calibration_vector():
    extractor, encoder = load_model()
    REFERENCE_FONT = "Yomogi-Regular"
    deltas = []  # (User - Font)

    font_folder = GENERATED_EMBEDDINGS_FOLDER / REFERENCE_FONT

    for image_file in CALIBRATION_IMAGES_FOLDER.glob("*.png"):
        user_image = Image.open(image_file, 'r')
        user_tensor = get_embeddings(extractor, encoder, [user_image])
        font_embed = torch.load(font_folder / f"{image_file.stem}.pt", weights_only=True)
        deltas.append(user_tensor[0] - font_embed)

    torch.save(torch.mean(torch.stack(deltas), 0), CALIBRATION_FILE)


def upload_embeddings():
    qdrant = create_connection()
    assert create_collection(qdrant), "Failed to create collection"

    standard_kanji_set = get_standard_kanji_set()
    font_tensors_folders = list(GENERATED_EMBEDDINGS_FOLDER.iterdir())
    print(f"Uploading embeddings for the following fonts: {tuple(folder.name for folder in font_tensors_folders)}")

    for folder in tqdm(font_tensors_folders):
        tensors = list(folder.glob("*.pt"))
        print(f"Found {len(tensors)} tensors for font {folder.name}")

        for file in tqdm(tensors):
            tensor = torch.load(file, weights_only=True)
            labels = file.with_suffix(".txt").read_text("UTF-8").splitlines()
            embeddings = {label: embedding for label, embedding in zip(labels, tensor)}

            insert(qdrant, folder.name, embeddings, standard_kanji_set)


def _search_files(files: list[Path]):
    qdrant = create_connection()
    extractor, encoder = load_model()

    if CALIBRATION_FILE.is_file():
        print("Loading Calibration file")
        calibration_vector = torch.load(CALIBRATION_FILE, weights_only=True)
        # An offset to approach the user's embeddings to that of the font
        # (Ideally you would probably want to try something akin to a lora,
        # but this worked descently for me and is very light to "train")
    else:
        calibration_vector = torch.zeros(MODEL_EMBEDDING_SIZE)

    images = [Image.open(file, "r").convert("L") for file in files]
    tensor = get_embeddings(extractor, encoder, images)

    for file, vector in zip(files, tensor):
        results = search_vector(qdrant, (vector - calibration_vector), limit=50)

        formatted = format_search_results(results)
        print(f"Search Results for {file.stem}:")
        print('\t'.join(dict.fromkeys(result.kanji for result in formatted)), end='\n')


def search_path(path: Path):
    if path.is_file():
        _search_files([path])
    elif path.is_dir():
        _search_files(list(path.glob("*.png")))
    else:
        raise Exception(f'Could not find a file nor a folder at Path "{path.resolve()}"')


if __name__ == "__main__":
    from cli import parser
    args = parser.parse_args()
    functions = {
        "generate_images": generate_images,
        "generate_embeddings": generate_embeddings,
        "upload_embeddings": upload_embeddings,
        "calibrate": create_calibration_vector,
        "search": lambda : search_path(args.input),
    }
    functions[args._name]()
