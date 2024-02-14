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

#### Downloading the data required to run it

The Kanji character lists available under https://gist.github.com/etrotta/ef2051b1168d878182950dcdce9e5d13
The fonts are available on Google Fonts

You have to download them yourself and store them in the paths specified by the `config.py` file, though you can freely change those paths as long as they contain the data necessary.
`INPUT_FONTS_FOLDER` Should fontain one subfolder per font, each including one or more `.ttf` files
`INPUT_KANJI_FOLDER` Should contain `.txt` files, each file containing one character per line

Running this project for a language other than Japanese would almost definitely require using a different model for the embeddings. (Maybe except Mandarin)

#### Setting up the database

Set the `QDRANT_URL` and `QDRANT_API_KEY` environment variables if you want to use Qdrant Cloud, or leave it empty if you are running one locally (e.g. via Docker)

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

### Accuracy
Far from 100%, but feels pretty good for me personally. There are no official benchmarks.

#### Where are the Fonts / Embeddings / Database
You have to download the character lists and fonts, then generate the Embeddings and upload to a Qdrant Database yourself using the instructions provided above.

At least as of this commit, the embeddings are not publicly available, but I can provide a snapshot of the database or a zip of the embeddings if you contact me.

(I don't mind making it public, but GitHub does not feels like the right place for it, and I don't know where to put it)

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
Not final yet, may still try a few different ones.

At first I considered looking for an Autoencoder, but couldn't find any that looked like they might work.

The model I ended up using, ``kha-white/manga-ocr``, was one of the few open source models tuned for Japanese text in vision tasks I could find, so I gave it a try and sticked with it after getting results I'm content with.


#### Vector Database
Using a vector database at all: About half of the reason why I decided to start this project at all was to get some experience with pytorch, transformers and vector databases

Specifically using `Qdrant`: Free, open source, easy to setup locally, the API looks nicer than other options I considered, free cloud tier.

### User specific Tuning / Calibration
Very experimental and rather rough/simplistic,

Given a list of user-created images and their known reference embeddings, take the average difference between the user's embeddings and the font's, then save that difference and add it when searching later.

## Acknowledgements
Kanji character lists:
- kanji_extra.txt (standard 2000ish) generated from https://www.kanjidatabase.com/
- kanji_joyo.txt (comprehensive list of >10k, filtered to exclude standard) generated from http://www.edrdg.org/wiki/index.php/KANJIDIC_Project

All fonts downloaded from Google Fonts, namely:
- DotGothic16
- SawarabiMincho
- Yomogi
- ZenKurenaido
- M_PLUS_1p
- Zen_Maru_Gothic
- Kaisei_Tokumin

It should work perfectly fine with different fonts though, as long as they support the characters you need.

It is intentional for neither of the character lists, the fonts nor the models to be included in this repository, in order to keep it as free of clutter as possible.

Model used for embedding: https://github.com/kha-white/manga-ocr

Vector database used: `qdrant`

Other tools used: see `requirements.txt`
