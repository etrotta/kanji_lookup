import os
import pathlib

ROOT = pathlib.Path.cwd()

DATA = ROOT / "data"

INPUTS = DATA / "inputs"
GENERATED = DATA / "generated"
TESTS = DATA / "test"

# Input: Fonts and text to generated images off

INPUT_FONTS_FOLDER = INPUTS / 'fonts'
INPUT_KANJI_FOLDER = INPUTS / 'text'

# Generated by the model: Images based off the input + Embeddings generated by the encoding model

GENERATED_IMAGES_FOLDER = GENERATED / 'images'
GENERATED_EMBEDDINGS_FOLDER = GENERATED / 'embeddings'

# Store the Model itself (it is already cached by Transformers, but I'd rather have it in the project folder)
# We also discard part of it, namely the Decoder that turns the ViT embeddings into text for the original ocr model

MODELS_FOLDER = ROOT / 'models'

EXTRACTOR_MODEL_PATH = MODELS_FOLDER / "extractor"
ENCODER_MODEL_PATH = MODELS_FOLDER / "encoder"

# Calibration

CALIBRATION_IMAGES_FOLDER = ROOT / '.testing' / 'calibrate'
CALIBRATION_FILE = ROOT / '.testing' / 'calibrate' / 'offset.pt'

MODEL = "kha-white/manga-ocr-base"

MODEL_EMBEDDING_SIZE = 768
MODEL_IMAGE_SIZE = 224
# Some sizes to try depending on the model: 96, 120, 184, 280
FONT_SIZE = 184

# V can set to `:memory:`, `localhost`, a file, or a cloud URL - see the qdrant docs for more info
DATABASE_LOCATION = os.getenv("QDRANT_URL", 'localhost')
DATABASE_API_KEY = os.getenv("QDRANT_API_KEY")
