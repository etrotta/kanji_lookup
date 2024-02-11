import pathlib
import numpy as np
from PIL import Image, ImageFont, ImageDraw

root = pathlib.Path.cwd()

IMAGES_FOLDER = root / 'images'
FONTS_FOLDER = root / 'fonts'

KANJI_LIST = (root / 'kanji.txt').read_text('UTF-8').splitlines()

# For testing, only do the first few + some hardcoded
KANJI_LIST = KANJI_LIST[:30] + list("猫狐狼四匹")

IMAGE_SIZE = 224
FONT_SIZE = 184  # sizes to try depending on the model: 96, 120, 184, 280


def draw_kanji(font: ImageFont.FreeTypeFont, kanji: str):
    "Create an image with the given `kanji` drawn in the given `font`"
    image = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((IMAGE_SIZE // 2, IMAGE_SIZE // 2), kanji, font=font, anchor="mm", fill=0)
    return image

def check_has_text(image: Image):
    "Verifies if the image contain anything at all"
    _arr = np.asarray(image)
    if _arr.min() == _arr.max():
        return False
    return True


for font_file in FONTS_FOLDER.glob("**/*.ttf"):
    font = ImageFont.truetype(font_file, FONT_SIZE)
    out_folder = IMAGES_FOLDER / font_file.stem
    out_folder.mkdir(exist_ok=True)
    for kanji in KANJI_LIST:
        out_file = out_folder / (kanji + '.png')
        image = draw_kanji(font, kanji)
        if not check_has_text(image):
            print(f"Font {font_file.stem} does not seems to support {kanji}, skipping it for this font")
            continue
        image.save(out_file)
