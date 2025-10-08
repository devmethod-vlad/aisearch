import contextlib
import hashlib
import os
import re
from pathlib import Path

import unicodedata

import nltk
import numpy as np
import pymorphy3


def configure_nltk_paths() -> Path:
    """Конфигурация пути nltk ресурсов"""
    base = Path(os.getenv("NLTK_DATA", "/srv/nltk_data"))  # общий путь в контейнере
    p = base.resolve()
    if str(p) not in nltk.data.path:
        nltk.data.path.insert(0, str(p))
    return p


def assert_nltk_resources_present() -> list[str]:
    """Проверка наличия необходимых ресурсов (возвращает отсутствующие)"""
    needed = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }
    missing = []
    for name, path in needed.items():
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)
    return missing


def download_nltk_resources(resources: list[str]) -> None:
    for recource in resources:
        nltk.download(recource)


def init_nltk_resources() -> None:
    """Загрузка ресурсов NLTK при инициализации приложения"""
    configure_nltk_paths()
    missing = assert_nltk_resources_present()
    if missing:
        download_nltk_resources(missing)
    missing = assert_nltk_resources_present()
    if missing:
        raise RuntimeError(
            f"Не удалось установить NLTK-ресурсы: {', '.join(missing)} " f"в {nltk.data.path!r}"
        )


def hash_query(normalized_query: str) -> str:
    """Хеширование нормализованного запроса"""
    return hashlib.md5(normalized_query.encode("utf-8")).hexdigest()


def normalize_query(query: str) -> str:
    """Нормализация запроса"""
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    morph = pymorphy3.MorphAnalyzer()
    stop_words = set(stopwords.words("russian"))
    text = query.strip().replace("\xa0", " ").replace("\n", " ")

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = unicodedata.normalize("NFC", text)

    tokens = word_tokenize(text)

    normalized_tokens = []
    for token in tokens:

        if not token.isalnum() or token in stop_words:
            continue

        lemma = morph.parse(token)[0].normal_form
        normalized_tokens.append(lemma)

    unique_tokens = sorted(set(normalized_tokens))
    return " ".join(unique_tokens)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Нормализация вектора"""
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n
