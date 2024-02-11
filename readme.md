## Kanji Lookup

Look up Kanji by drawing them

WIP

### Project Overview
- Generate synthetic Kanji images using multiple different fonts
- Encode into Embeddings using a Neural Network
- Store the Embeddings in a Vector Database (TODO)
- Encode an User submitted image and compare it with existing records (TODO)

In an ideal world:
- gradio app to draw and lookup in real time?


### Reasoning

#### Inspiration for this project
I was annoyed at how bad `jisho.org` Draw based searching is, so I decided to try and do it better myself.
Don't get me wrong, it is an amazing website, and it's possible that it works great if you know how to use it well (get the kanji stroke count, order, etc. right), but it really does not works for me.

#### Synthetic images over handwritten
Couldn't find any handwritten datasets I liked
(most free ones look way too low quality, I did not look into paid ones at all)

#### Fonts used
Literally just looked up `Japanese Fonts` on Google then found [Google Fonts](https://fonts.google.com/) and search in it, then picked some of the first ones that looked promising.
I tried to choose some different styles and (for most part) closer to handwritting than rigid shapes, but really didn't think much about that.

#### Model used for the embedding
Not final yet, will still try a few different ones
At first I considered an Autoencoder or something akin to OpenAI CLIP's, but couldn't find any that looked like they might work
The most promising models in my inexperienced view were ``kha-white/manga-ocr``, which I am currently experimenting with, and `google/deplot`, which I may test later if I cannot get good enough results out of manga-ocr.

#### Vector Database
Using a vector database at all: About half of why I decided to work on this at all was to get some experience with pytorch, transformers and vector databases
Specifically `milvus`: Free, open source, easy to setup locally, one of the first three that I looked into (the other two being cloud first, if not cloud-only).

## Acknowledgements
kanji.txt generated from https://www.kanjidatabase.com/

All fonts downloaded from Google Fonts, namely:
- DotGothic16
- SawarabiMincho
- Yomogi
- ZenKurenaido
(Note: It is intentional for them to not be included in the repository. Download them and add their `.ttf` files to the `/fonts` folder to run it yourself.)
Still plan to add and test multiple other fonts


Model used for embedding: https://github.com/kha-white/manga-ocr
(TODO TEST `GOOGLE/DEPLOT` MODEL?)


Vector database used: `milvus`
(note: as of this commit, not in use still, just planning to use)

Other tools used: see `requirements.txt`
