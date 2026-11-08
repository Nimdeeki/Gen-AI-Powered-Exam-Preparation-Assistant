"""Text preprocessing, following the steps from the class Colab notebook:
lowercasing -> remove HTML tags -> remove URLs -> remove punctuation ->
tokenization -> stop-word removal -> stemming / lemmatization.
"""

import re
import string
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


@lru_cache(maxsize=1)
def _ensure_nltk_data() -> None:
    for resource, package in [("corpora/stopwords", "stopwords"),
                              ("tokenizers/punkt_tab", "punkt_tab"),
                              ("corpora/wordnet", "wordnet")]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def stopword_set() -> frozenset[str]:
    _ensure_nltk_data()
    from nltk.corpus import stopwords
    return frozenset(stopwords.words("english"))


_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


def remove_html_tags(text: str) -> str:
    return re.sub(r"<.*?>", " ", text)


def remove_url(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_punc(text: str) -> str:
    return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))


def tokenize(text: str) -> list[str]:
    _ensure_nltk_data()
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    stops = stopword_set()
    return [t for t in tokens if t not in stops]


def stem(tokens: list[str]) -> list[str]:
    return [_stemmer.stem(t) for t in tokens]


@lru_cache(maxsize=50_000)
def lemma(word: str) -> str:
    """Noun lemma first ("transformers" -> "transformer"), then verb ("running" -> "run")."""
    _ensure_nltk_data()
    noun = _lemmatizer.lemmatize(word, pos="n")
    return noun if noun != word else _lemmatizer.lemmatize(word, pos="v")


def lemmatize(tokens: list[str]) -> list[str]:
    return [lemma(t) for t in tokens]


def preprocess(text: str) -> list[str]:
    """Full pipeline: returns clean, lemmatized content-word tokens."""
    text = remove_punc(remove_url(remove_html_tags(text.lower())))
    tokens = [t for t in tokenize(text) if t.isalpha() and len(t) > 1]
    return lemmatize(remove_stopwords(tokens))


def pipeline_report(text: str) -> dict:
    """Numbers for every preprocessing step, shown in the app's 'NLP pipeline' view."""
    lowered = text.lower()
    no_html = remove_html_tags(lowered)
    no_url = remove_url(no_html)
    no_punc = remove_punc(no_url)
    tokens = [t for t in tokenize(no_punc) if t.isalpha()]
    content = remove_stopwords(tokens)
    stems, lemmas = stem(content), lemmatize(content)
    sample = [(w, s, l) for w, s, l in zip(content, stems, lemmas) if w != s or w != l]
    return {
        "characters": len(text),
        "tokens": len(tokens),
        "vocabulary": len(set(tokens)),
        "after_stopwords": len(content),
        "vocab_after_stemming": len(set(stems)),
        "vocab_after_lemmatization": len(set(lemmas)),
        "stem_vs_lemma_examples": list(dict.fromkeys(sample))[:12],
    }
