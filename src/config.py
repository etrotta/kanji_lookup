import pathlib

ROOT = pathlib.Path.cwd()

IMAGES_FOLDER = ROOT / 'images'
FONTS_FOLDER = ROOT / 'fonts'
MODELS_FOLDER = ROOT / 'models'
EMBEDDINGS_FOLDER = ROOT / 'embeddings'

EXTRACTOR_PATH = MODELS_FOLDER / "extractor"
ENCODER_PATH = MODELS_FOLDER / "encoder"

MODEL = "kha-white/manga-ocr-base"

MODEL_EMBEDDING_SIZE = 768
MODEL_IMAGE_SIZE = 224
# Some sizes to try depending on the model: 96, 120, 184, 280
FONT_SIZE = 184

DATABASE_LOCATION = 'localhost'  # can set to `:memory:`, `localhost`, a file, or a cloud URL - see the qdrant docs for more info
