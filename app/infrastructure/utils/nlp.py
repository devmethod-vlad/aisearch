import contextlib
import hashlib
import re
import unicodedata

import nltk
import pymorphy3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def download_nltk_resources() -> None:
    """Загрузка ресурсов NLTK при инициализации приложения"""
    required_resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }
    for resource_name, resource_path in required_resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            with contextlib.suppress(FileExistsError):
                nltk.download(resource_name)


# temporal
def get_documents() -> list[str]:
    """Получить тестовый набор документов"""
    return [
        "Машинное обучение — это область искусственного интеллекта.",
        "Нейронные сети используются для обработки больших объемов данных.",
        "Python — популярный язык программирования для анализа данных.",
        "Русский язык — один из самых сложных языков мира.",
    ]


def hash_query(normalized_query: str) -> str:
    """Хеширование нормализованного запроса"""
    return hashlib.md5(normalized_query.encode("utf-8")).hexdigest()


def normalize_query(query: str) -> str:
    """Нормализация запроса"""
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
