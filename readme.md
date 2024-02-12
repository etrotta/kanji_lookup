## Kanji Lookup

Look up Kanji by drawing them

WIP

### Project Overview
- Generate synthetic Kanji images using multiple different fonts
- Encode into Embeddings using a Neural Network
- Store the Embeddings in a Vector Database
- Encode an User submitted image and compare it with existing records

Not implemented yet, but considering:
- gradio app to draw and lookup in real time?

### Running it

Download Japanese fonts and extract them to the a /fonts folder.

Create a `224x224` greyscale image called "test.png" on the repository root folder containing a Kanji drawing, or create a "drawings" folder and fill in with however many images you want.
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

### Accuracy
Far from 100%, but honestly better than what I expected. From my limited testing (see: /drawings), most were found within the top 5 search results, and only two completely missed top 20

(The ones which missed were 英 and 印, which were very poorly drawn so makes sense. The versions in /drawing_second_shot work though.

I kept them as the original version in /drawings to show some failure cases and make it clear it isn't perfect.)

Note: I only populated the first 200 kanji x 4 fonts for this test, the accuracy may lower once it has more data in the database.

#### Where are the Fonts / Embeddings / Database
You have to download the Fonts, generate the Embeddings and upload to a Qdrant Database yourself using the commands provided above.

At least as of this commit, the embeddings are not publicly available, but I can provide a dump of the database or a zip of the embeddings if you contact me.

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

At first I considered an Autoencoder or something akin to OpenAI CLIP's, but couldn't find any that looked like they might work.

The most promising models in my inexperienced view were ``kha-white/manga-ocr``, which I am currently using, and `google/deplot`, which I may test later if I cannot get good enough results out of `manga-ocr`.

Currently seems like `manga-ocr` works just fine though.

#### Vector Database
Using a vector database at all: About half of the reason why I decided to start this project at all was to get some experience with pytorch, transformers and vector databases

Specifically using `Qdrant`: Free, open source, easy to setup locally, the API looks nicer than other options, free cloud tier.

#### Distance Metric
I'm using `cosine similarity` just because it is the one I had heard about in the context of vector search before. Literally just that.

I might test the other types and compare the results, but did not research in depth.

## Acknowledgements
kanji.txt generated from https://www.kanjidatabase.com/
* I may still look for other sources later, to include common non-standard Kanji

All fonts downloaded from Google Fonts, namely:
- DotGothic16
- SawarabiMincho
- Yomogi
- ZenKurenaido

(Note: It is intentional for them to not be included in the repository. Download them and extract to the `/fonts` folder to run it yourself.)

I still plan to add and test multiple other fonts

Model used for embedding: https://github.com/kha-white/manga-ocr

Vector database used: `qdrant`

Other tools used: see `requirements.txt`
