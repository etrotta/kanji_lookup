## Kanji Lookup

Look up Kanji by drawing them

You can access an online demo in https://huggingface.co/spaces/etrotta/kanji_lookup

### Project Overview
- Generate synthetic Kanji images using multiple different fonts
- Encode into Embeddings using a Neural Network
- Store the Embeddings in a Vector Database
- Encode an User submitted image and compare it with existing records

In addition to that, I also added an experimental support to "tune" the search to an specific user's drawings, through the calibration defined in `src/main.py` (Not supported in the HuggingFace Space)

### Running it

#### Downloading the Embeddings

You can download a parquet file containing the embeddings from Hugging Face Datasets, https://huggingface.co/datasets/etrotta/kanji_embeddings/blob/main/kanji_embeddings.parquet 


After downloading them, you can use the `dataset/upload.py` file from this repository to upload it to a Qdrant database.


(`dataset/main.py` is used for generating the parquet file containing the embeddings. `dataset/test.py` is used for manually testing them. `src/*` are used for generating embeddings, as well as uploading and searching them. The `main.py upload_embeddings` command expects a format different from the parquet file though.)


Note: As of this commit, the datasets library does not supports it because it uses the Apache Arrow equivalent of an Enum.

#### Generating the Embeddings

Alternatively, you can generate the Embeddings yourself - feel free to mix and match different models, different character lists, langauges and 
fonts, though you might have to edit some things yourself to get it running for different models.


The Kanji character lists I used are available under https://gist.github.com/etrotta/ef2051b1168d878182950dcdce9e5d13 and you can find the fonts in Google Fonts.


You have to download them yourself and store them in the paths specified by the `config.py` file, though you can freely change those paths as long as they contain the data necessary.
- `INPUT_FONTS_FOLDER` Should fontain one subfolder per font, each including one or more `.ttf` files
- `INPUT_KANJI_FOLDER` Should contain `.txt` files, each file containing one character per line

Running this project for a language other than Japanese may require using a different model for the embeddings.

#### Setting up the database

Set the `QDRANT_URL` and `QDRANT_API_KEY` environment variables if you want to use Qdrant Cloud, or leave it empty if you are running one locally (e.g. via Docker)

If you're using the public dataset,
```
pip install -r dataset/requirements.txt

py dataset/upload.py
```

If you want to generate your own weights,
```
pip install -r requirements.txt

py src/main.py generate_images
py src/main.py generate_embeddings
py src/main.py upload_embeddings
```
Alternatively, you can use the notebook to generate the images and Embeddings, then use main.py to upload_embeddings

#### Searching based on image files

You can search either by individual images or by entire folders at a time.

Ideally the images should be `224x224` greyscale, containing exactly one character each file.

```
py src/main.py search test.png
py src/main.py search path/to/drawings_folder
```

#### Where are the Fonts / Embeddings / Database
You have to either download the Embeddings from Hugging Face Datasets, or download both the character lists and fonts then generate the Embeddings yourself.

One way or the other, after getting Embeddings, you should upload them to a Qdrant Database using the instructions provided above.

The embeddings are available under https://huggingface.co/datasets/etrotta/kanji_embeddings/blob/main/kanji_embeddings.parquet


*Note: The HuggingFace official `datasets` library may not support it. Use polars or pandas to read it instead.

## About the project itself

#### Inspiration
I was annoyed at how bad `jisho.org` Draw based searching is, so I decided to try and do it better myself.
Don't get me wrong, it is an amazing website, and it's possible that it works great if you know how to use it well (get the kanji stroke count, order, etc. right), but it really does not works for me.

#### Using Synthetic images over handwritten
Couldn't find any handwritten datasets I liked
(most free ones look way too low quality, I did not look into paid ones at all)

#### Fonts used
Literally just looked up `Japanese Fonts` on Google and found [Google Fonts](https://fonts.google.com/), then picked some of the first ones that looked promising.
I tried to choose some different styles and (for most part) closer to handwritting than rigid shapes, but really didn't think much about that.

## Technical details

#### Model used for the embedding
At first I considered looking for an Autoencoder, but couldn't find any that looked like they might work.

The model I ended up using, ``kha-white/manga-ocr``, was one of the few open source models tuned for Japanese text in vision tasks I could find, so I gave it a try and sticked with it after getting results I'm content with.


#### Vector Database
Using a vector database at all: About half of the reason why I decided to start this project at all was to get some experience with pytorch, transformers and vector databases

Specifically using `Qdrant`: Free, open source, easy to setup locally, the API looks nicer than other options I considered, free cloud tier.

#### User specific Tuning / Calibration
Very experimental and rather rough/simplistic,

Given a list of user-created images and their known reference embeddings, take the average difference between the user's embeddings and the font's, then save that difference and add it when searching later.

## Acknowledgements
Kanji character lists:
- kanji_joyo.txt (standard 2000ish) generated from https://www.kanjidatabase.com/
- kanji_non_joyo.txt (comprehensive list of >10k, filtered to exclude standard) generated from http://www.edrdg.org/wiki/index.php/KANJIDIC_Project

All fonts downloaded from Google Fonts, namely:
- DotGothic16
- SawarabiMincho
- Yomogi
- ZenKurenaido
- M_PLUS_1p
- Zen_Maru_Gothic
- Kaisei_Tokumin

It should work perfectly fine with different fonts though, as long as they support the characters you need.

It is intentional for neither of the character lists, the fonts, the embeddings nor the models to be included in this repository, in order to keep it as free of clutter as possible.

Model used for embedding: https://github.com/kha-white/manga-ocr
(In particular, the ViT Model encoder part)

Vector database used: https://qdrant.tech/
