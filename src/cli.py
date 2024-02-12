from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# (generate images, generate embeddings, upload to database, search user drawing)

# IMAGES
arg_generate_images = subparsers.add_parser("generate_images")
arg_generate_images.set_defaults(_name="generate_images")

# arg_generate_images.add_argument("--file", default="kanji.txt", type=Path)

# _fonts_group = arg_generate_images.add_mutually_exclusive_group()
# _fonts_group.add_argument("--font", type=Path)
# _fonts_group.add_argument("--fonts-folder", default="fonts", type=Path)

# arg_generate_images.add_argument("-o", "--output", default="images", type=Path)

# EMBEDDINGS
arg_generate_embeddings = subparsers.add_parser("generate_embeddings")
arg_generate_embeddings.set_defaults(_name="generate_embeddings")

# arg_generate_embeddings.add_argument("--input", default="images", type=Path)

# arg_generate_embeddings.add_argument("--base-model", default="manga-ocr")  # Unused (for now?)

# arg_generate_embeddings.add_argument("-o", "--output", default="embeddings", type=Path)

# CREATE DATABASE
arg_populate_database = subparsers.add_parser("populate_database")
arg_populate_database.set_defaults(_name="populate_database")

# arg_populate_database.add_argument("--input", default="embeddings", type=Path)

# arg_populate_database.add_argument("--database-location")

# SEARCH DATABASE
arg_search = subparsers.add_parser("search")
arg_search.set_defaults(_name="search")

arg_search.add_argument("input", type=Path)
# arg_search.add_argument("--database-location")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)


