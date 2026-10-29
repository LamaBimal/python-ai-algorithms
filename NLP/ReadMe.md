# Nepali news classification

`NLPWorkshop.ipynb` is a completed teaching walkthrough;
`NepaliNewsClassification.ipynb` is its executable reference. Both share the
tested implementation in `news_classification.py`.

From the repository root, install `requirements.txt` and run
`python -m notebook`. Select that environment's kernel and run all cells in
order. You can start Jupyter from the root or the `NLP` directory.

The notebooks use the included UTF-8 `data.csv` and `suffix.txt`. Data files
are stored with Git LFS; use `git lfs install` and `git lfs pull` after cloning
if the checkout contains pointer files instead of data.

The workflow is:

1. Keep Nepali news from sports, business, entertainment, and domestic news and remove duplicate
   articles. Sample a bounded candidate pool before text preprocessing, then
   deduplicate cleaned text and balance the final sample to at most 200 per topic.
2. Clean Devanagari text, tokenize on whitespace, remove a small illustrative
   stopword list, and apply a simple longest-suffix-first stemmer.
3. Make a fixed stratified train/test split and fit TF-IDF on training rows only.
4. Evaluate multiclass logistic regression with accuracy, per-class metrics,
   and a confusion matrix.
5. Compare a feed-forward neural network using the same split.
6. Clean and classify a new article using the fitted pipeline.

TF-IDF replaces the original pretrained FastText vectors so a normal CPU can
run the full example without external downloads, NLTK corpora, or PyTorch.
It provides word-frequency features rather than pretrained semantic embeddings.
The stemmer and stopwords are teaching heuristics. Scores describe a small,
balanced sample and should not be interpreted as population performance.

Run `python -m unittest discover -s tests -v` after installing
`requirements-dev.txt` to execute both notebooks and their regression checks.
