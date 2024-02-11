import pathlib
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from transformers import (
    # AutoFeatureExtractor,  # Load extractor from Hub
    VisionEncoderDecoderModel,  # Load ViT model from Hub
    # ViTFeatureExtractor,  # Load local extractor
    ViTImageProcessor,  # Load local extractor
    ViTModel,  # Load local ViT encoder
)

MODEL = "kha-white/manga-ocr-base"

ROOT = pathlib.Path.cwd()

MODELS_FOLDER = ROOT / 'models'

EXTRACTOR_PATH = MODELS_FOLDER / "extractor"
ENCODER_PATH = MODELS_FOLDER / "encoder"

if EXTRACTOR_PATH.is_dir():
    print("Loading local Image Processor")
    feature_extractor = ViTImageProcessor.from_pretrained(EXTRACTOR_PATH)
else:
    # feature_extractor: ViTFeatureExtractor = AutoFeatureExtractor.from_pretrained(MODEL)
    # ^ Deprecated
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
    del model
    encoder.save_pretrained(ENCODER_PATH)

# Note: The original repository also features a tokenizer,
# But we do not use it at all (it is used to turn the decoded text back from tokens into unicode)

# TODO Test removing the last activation layer of the encoder?

# Side note: The original models also remain in your Hugging Face default `.cache` folder
# (even more offtopic) ...and the venv is pretty heavy


# ---

IMAGES_FOLDER = ROOT / "images" 

FONT_IMAGES_PATH = IMAGES_FOLDER / "Yomogi-Regular"
TEST_KANJI = "猫狐狼四匹"

images = {
    kanji: Image.open(FONT_IMAGES_PATH / f"{kanji}.png", 'r')
    for kanji in TEST_KANJI
}

# ---

# next(iter(images.values())).show()

# pre-processing
_images = [images[kanji].convert("RGB") for kanji in TEST_KANJI]
pixel_values: torch.Tensor = feature_extractor(_images, return_tensors="pt")["pixel_values"].squeeze()

# get the ViT embedding
out = encoder(pixel_values)["pooler_output"]

embeddings = {
    kanji: out[i]
    for i, kanji in enumerate(TEST_KANJI)
}

cat, fox, wolf, four, animals = TEST_KANJI

def calc(vec_a: torch.Tensor, vec_b: torch.Tensor):
    _vec_a = (vec_a * 0.5) + 0.5
    _vec_b = (vec_b * 0.5) + 0.5
    return cosine_similarity(_vec_a, _vec_b, dim=0)

tests_cases = [
    (cat, fox),
    (cat, wolf),
    (wolf, fox),

    (cat, four),
    (cat, animals),
    (fox, four),
    (fox, animals),

    (animals, four)
]
for test_case in tests_cases:
    print(test_case, calc(embeddings[test_case[0]], embeddings[test_case[1]]))

# ---

TEST_IMAGE = Image.open(ROOT / "tmp.png", 'r')

# TEST_IMAGE.show()

# pre-processing
_test_pixel_values: torch.Tensor = feature_extractor(TEST_IMAGE.convert("RGB"), return_tensors="pt")["pixel_values"].squeeze()

# get the ViT embedding

_test_out = encoder(_test_pixel_values[None])["pooler_output"][0]

print(dict(sorted({kanji: calc(_test_out, embeddings[kanji]) for kanji in TEST_KANJI}.items(), key=lambda t: t[1], reverse=True)))
