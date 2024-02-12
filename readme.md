## Kanji Lookup

Look up Kanji by drawing them

WIP

## Running it

Download fonts and extract them to the a /fonts folder
Create a 224x224 greyscale image called "test.png" on the repository root folder containing a Kanji drawing, or create a "drawings" folder and fill in with however many images you want.
```
pip install -r requirements.txt

py src/main.py generate_images
py src/main.py generate_embeddings
py src/main.py populate_database
```
Then for searching, you can choose either one individual file or an entire folder:
```
py src/main.py search test.png
py src/main.py search drawings
```

#### Accuracy
Far from 100%, but honestly better than what I expected, from my limited testing (see: /drawings), most were found amongst top 5, and only two completely missed top 20 (英 and 印, which are honestly *that* poorly drawn so makes sense - The versions in /drawing_second_shot works though. I kept them as the original version in the normal folder to show some failure cases and not pretend it is perfect.)

Note: I only populated the first 200 kanji x 4 fonts yet though

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
Specifically `qdrant`: Free, open source, easy to setup locally, the API looks nicer than other options, free cloud tier.

#### Index Metric Type
I'm using `cosine similarity` just because it is the one I had heard about in the context of vector search before. Literally just that.

I might test the other types and compare the results, but did not research it at all.

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


Vector database used: `qdrant`

Other tools used: see `requirements.txt`
