from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,  # Load extractor
    ViTModel,  # Load ViT encoder
)

from config import (
    MODEL,
    MODEL_EMBEDDING_SIZE,
    EXTRACTOR_MODEL_PATH,
    ENCODER_MODEL_PATH,
)

# Didn't want to hardcode within this file, and may have to use elsewhere
assert MODEL == "kha-white/manga-ocr-base", "Other models are not natively supported, \
    you may have to change a lot of things to get it to work"
assert MODEL_EMBEDDING_SIZE == 768, "The only model embedding size supported is 768"

def load_model() -> tuple[ViTImageProcessor, ViTModel]:
    """Load the model based on the config.py file.
    Returns the `feature_extractor` and the `encoder`, in this order.
    """
    if EXTRACTOR_MODEL_PATH.is_dir():
        print("Loading local Image Processor")
        feature_extractor = ViTImageProcessor.from_pretrained(EXTRACTOR_MODEL_PATH, requires_grad=False)
    else:
        print("Loading Image Processor from HuggingFace Hub")
        feature_extractor: ViTImageProcessor = ViTImageProcessor.from_pretrained(MODEL, requires_grad=False)
        feature_extractor.save_pretrained(EXTRACTOR_MODEL_PATH)

    if ENCODER_MODEL_PATH.is_dir():
        print("Loading local ViT Encoder Model")
        model = ViTModel.from_pretrained(ENCODER_MODEL_PATH)
    else:
        print("Loading ViT Model from HuggingFace Hub")
        # base_model = ViTModel.from_pretrained(MODEL)
        model: ViTModel = VisionEncoderDecoderModel.from_pretrained(MODEL).encoder
        model.save_pretrained(ENCODER_MODEL_PATH)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
    else:
        print('Using CPU')

    return feature_extractor, model


def get_embeddings(feature_extractor: ViTImageProcessor, encoder: ViTModel, images: list[Image.Image]) -> torch.Tensor:
    """Processes the images and returns their Embeddings"""
    images_rgb = [image.convert("RGB") for image in images]
    with torch.inference_mode():
        pixel_values: torch.Tensor = feature_extractor(images_rgb, return_tensors="pt")["pixel_values"]
        return encoder(pixel_values.to(encoder.device))["pooler_output"].cpu()


def compare_vectors(vec_a: torch.Tensor, vec_b: torch.Tensor):
    # Note: Not actually used outside of the `if __name__ == "__main__":` test, since we are using a vector database
    _vec_a = (vec_a * 0.5) + 0.5
    _vec_b = (vec_b * 0.5) + 0.5
    return cosine_similarity(_vec_a, _vec_b, dim=0)


if __name__ == "__main__":
    from config import ROOT
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
