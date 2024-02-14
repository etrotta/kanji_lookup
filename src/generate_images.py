import numpy as np
from PIL import Image, ImageFont, ImageDraw

from config import (
    INPUT_KANJI_FOLDER,
    INPUT_FONTS_FOLDER,
    MODEL_IMAGE_SIZE,
    FONT_SIZE,
)

def get_standard_kanji_set() -> set[str]:
    file = INPUT_KANJI_FOLDER / "kanji_joyo.txt"
    return set(file.read_text(encoding="UTF-8").splitlines())

def load_kanji_list() -> list[str]:
    kanji_list = []
    for file in INPUT_KANJI_FOLDER.glob("*.txt"):
        kanji_list += file.read_text(encoding="UTF-8").splitlines()
    return kanji_list


def draw_kanji(font: ImageFont.FreeTypeFont, kanji: str):
    "Create an image with the given `kanji` drawn in the given `font`"
    image = Image.new('L', (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((MODEL_IMAGE_SIZE // 2, MODEL_IMAGE_SIZE // 2), kanji, font=font, anchor="mm", fill=0)
    return image

def check_has_text(image: Image):
    "Verifies if the image contain anything at all"
    _arr = np.asarray(image)
    if _arr.min() == _arr.max():
        return False
    return True


def list_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    """Returns a dictionary of `font_name -> ImageFont`"""
    return {
        font_file.stem: ImageFont.truetype(font_file, FONT_SIZE)
        for font_file in INPUT_FONTS_FOLDER.glob("**/*.ttf")
    }

def generate_images_for_font(font: ImageFont.FreeTypeFont, kanji_list: list[str]) -> dict[str, Image.Image]:
    """Returns a dictionary of `kanji -> Image`"""
    out = {}
    for kanji in kanji_list:
        image = draw_kanji(font, kanji)
        if not check_has_text(image):
            print(f"Font {font} does not seems to support {kanji}, skipping it for this font")
            continue
        out[kanji] = image
    return out

if __name__ == "__main__":
    from config import ROOT
    font_name, font = list(list_fonts().items())[0]
    out_folder = ROOT / ".testing" / "images" / font_name
    out_folder.mkdir(exist_ok=True, parents=True)
    kanji_list = load_kanji_list()[:20]
    kanji_list += list("猫狐狼四匹")
    images = generate_images_for_font(font, kanji_list)
    for kanji, img in images.items():
        img.save(out_folder / f"{kanji}.png")
