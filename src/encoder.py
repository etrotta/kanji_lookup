from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from transformers import (
    VisionEncoderDecoderModel,  # Load ViT model from Hub
    ViTImageProcessor,  # Load local extractor
    ViTModel,  # Load local ViT encoder
)

from config import (
    ROOT,
    MODEL,
    EXTRACTOR_PATH,
    ENCODER_PATH,
)

assert MODEL == "kha-white/manga-ocr-base", "Other models are not natively supported, \
    you may have to change a lot of things to get it to work"


def load_model() -> tuple[ViTImageProcessor, ViTModel]:
    """Load the model based on the config.py file.
    Returns the `feature_extractor` and the `encoder`, in this order.
    """
    if EXTRACTOR_PATH.is_dir():
        print("Loading local Image Processor")
        feature_extractor = ViTImageProcessor.from_pretrained(EXTRACTOR_PATH)
    else:
        print("Loading Image Processor from HuggingFace Hub")
        feature_extractor: ViTImageProcessor = ViTImageProcessor.from_pretrained(MODEL)
        feature_extractor.save_pretrained(EXTRACTOR_PATH)

    if ENCODER_PATH.is_dir():
        print("Loading local ViT Encoder Model")
        encoder = ViTModel.from_pretrained(ENCODER_PATH)
    else:
        print("Loading ViT Model from HuggingFace Hub")
        model = VisionEncoderDecoderModel.from_pretrained(MODEL)
        encoder: ViTModel = model.encoder
        encoder.save_pretrained(ENCODER_PATH)

    return feature_extractor, encoder


# TODO Test removing the last activation layer of the encoder?


def get_embeddings(feature_extractor: ViTImageProcessor, encoder: ViTModel, images: list[Image.Image]) -> torch.Tensor:
    """Processes the images and returns their Embeddings"""
    _images = [image.convert("RGB") for image in images]
    pixel_values: torch.Tensor = feature_extractor(_images, return_tensors="pt")["pixel_values"].squeeze()

    return encoder(pixel_values)["pooler_output"]

def compare_vectors(vec_a: torch.Tensor, vec_b: torch.Tensor):
    _vec_a = (vec_a * 0.5) + 0.5
    _vec_b = (vec_b * 0.5) + 0.5
    return cosine_similarity(_vec_a, _vec_b, dim=0)


if __name__ == "__main__":
    FONT_IMAGES_PATH = next(iter((ROOT / ".testing" / "images").iterdir()))
    TEST_KANJI = "猫狐狼四匹"
    cat, fox, wolf, four, animals = TEST_KANJI

    extractor, encoder = load_model()

    images = {kanji: Image.open(FONT_IMAGES_PATH / f"{kanji}.png", "r") for kanji in TEST_KANJI}
    assert list(images.values()) == [images[k] for k in TEST_KANJI]
    test_image = Image.open(ROOT / ".testing" / "drawing.png", "r")

    tensor = get_embeddings(extractor, encoder, list(images.values()) + [test_image])
    embeddings = {name: tensor[i] for i, name in enumerate(TEST_KANJI)}
    drawing_embedding = tensor[len(TEST_KANJI)]

    tests_cases = [(cat, fox), (cat, wolf), (wolf, fox), (cat, four), (cat, animals), (fox, four), (fox, animals), (animals, four)]
    for test_case in tests_cases:
        print(test_case, compare_vectors(embeddings[test_case[0]], embeddings[test_case[1]]))

    _comparations = {kanji: compare_vectors(drawing_embedding, embeddings[kanji]) for kanji in TEST_KANJI}
    print(dict(sorted(_comparations.items(), key=lambda t: t[1], reverse=True)))
