import argparse
import asyncio
import gc
import importlib
import logging
import os
import traceback

import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


async def load_collection_and_index(
    settings: BaseSettings,
    logger: logging.Logger,
    collection_name: str = "kb_default",
    column_for_vector: str = "question",
) -> None:
    """Загрузить коллекцию в Milvus DB"""
    from app.infrastructure.storages.milvus import MilvusDatabase

    milvus = MilvusDatabase(settings=settings.milvus, logger=logger)

    logger.info(f"Выполняется загрузка модели {settings.milvus.model_name.split('/')[-1]} ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.milvus.model_name)
    logger.info("Модель успешно загружена")

    df = pd.read_parquet(f"{collection_name}.parquet")
    df = df[
        df[column_for_vector].notna() & (df[column_for_vector].astype(str).str.len() > 0)
    ].reset_index(drop=True)

    if settings.milvus.recreate_collection or settings.opensearch.recreate_index:
        from app.infrastructure.adapters.open_search import OpenSearchAdapter

        os_adapter = OpenSearchAdapter(settings=settings, logger=logger)
        os_adapter.build_index(data=df)

    documents = df[column_for_vector].tolist()

    if settings.milvus.recreate_collection or settings.bm25.recreate_index:
        from app.infrastructure.adapters.bm25 import BM25Adapter

        BM25Adapter.build_index(
            data=df, index_path=settings.bm25.index_path, texts=documents, logger=logger
        )

    await milvus.initialize_model_metadata_collection()
    try:
        await milvus.ensure_collection(
            collection_name=collection_name,
            model=model,
            documents=documents,
            metadata=df,
            recreate=settings.milvus.recreate_collection,
        )
    except Exception as e:
        logger.error(f"Произошла ошибка при загрузке коллекции: {e}")
        await milvus.delete_collection(collection_name=collection_name)

    await milvus.close()
    del model
    importlib.invalidate_caches()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def pre_launch() -> None:
    """Скрипт, запускающийся перед приложением"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--logtype", help="Тип логирования", choices=["app", "value"], default="app"
    )
    parser.add_argument("--load", help="Загрузка коллекции в Milvus DB", action="store_true")
    args = parser.parse_args()

    if args.local:
        os.environ["MILVUS_HOST"] = "localhost"
        os.environ["MILVUS_MODEL_NAME"] = os.environ["APP_MODELSTORE_HOST_PATH"] + "/LaBSE-ru-turbo"
        os.environ["RERANKER_MODEL_NAME"] = (
            os.environ["APP_MODELSTORE_HOST_PATH"] + "/cross-encoder-russian-msmarco"
        )
        os.environ["OS_HOST"] = "localhost"
        os.environ["BM25_INDEX_PATH"] = os.environ["BM25_INDEX_PATH_HOST"]

    from app.settings.config import Settings

    settings = Settings()

    from app.common.logger import AISearchLogger, LoggerType

    logger = AISearchLogger(logger_type=LoggerType(args.logtype))

    logger.info("Запуск приложения...")
    if args.load:
        logger.info("Загрузка данных в Milvus DB...")
        try:
            await load_collection_and_index(settings, logger)
        except Exception as e:
            logger.error(f"Ошибка при ожидании базы данных: ({type(e)}): {traceback.format_exc()}")
            raise e


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(pre_launch())
