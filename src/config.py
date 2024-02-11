import pathlib

ROOT = pathlib.Path.cwd()

IMAGES_FOLDER = ROOT / 'images'
FONTS_FOLDER = ROOT / 'fonts'
MODELS_FOLDER = ROOT / 'models'

EXTRACTOR_PATH = MODELS_FOLDER / "extractor"
ENCODER_PATH = MODELS_FOLDER / "encoder"

MODEL = "kha-white/manga-ocr-base"

MODEL_IMAGE_SIZE = 224
# Some sizes to try depending on the model: 96, 120, 184, 280
FONT_SIZE = 184


