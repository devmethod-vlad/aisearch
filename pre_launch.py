import argparse
import asyncio
import datetime
import logging
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from app.infrastructure.utils.prepare_dataframe import (
    dedup_by_question_any,
    prepare_dataframe,
    rename_dataframe
)
from app.infrastructure.utils.universal import cleanup_resources


def setup_logger(logtype: str) -> logging.Logger:
    """Настраивает и возвращает логгер для pre_launch скрипта."""
    prefix = "[pre_launch.py]"

    logger = logging.getLogger(logtype)

    original_handlers = logger.handlers.copy()
    for handler in original_handlers:
        logger.removeHandler(handler)

    logs_path = (
        os.getenv("CELERY_LOGS_PATH") if logtype == "celery" else os.getenv("LOG_PATH")
    )

    if not logs_path:
        raise ValueError(f"Переменная окружения для логов не установлена: {logtype}")

    log_file = Path(logs_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    class CustomFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            ts = datetime.datetime.now(tz=datetime.UTC).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            base = f"{ts} {prefix} [{record.levelname}] {record.getMessage()}"
            if record.exc_info:
                base += "\n" + self.formatException(record.exc_info)
            return base

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(CustomFormatter())
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(CustomFormatter())
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    logger.propagate = False

    return logger


def exit_with_error(logger: logging.Logger, text: str, code: int = 1) -> None:
    """Завершить работу скрипта с ошибкой"""
    logger.error(text)
    logger.info(f"Завершение pre_launch ({code})")
    sys.exit(code)


def prepare_data_from_file(
    file_path: str, id_field_name: str, logger: logging.Logger
) -> tuple:
    """Загружает данные из Parquet или Excel и подготавливает их для обработки.
    Фильтрует по наличию текста в колонках и условию для пользователя.
    """
    import pandas as pd

    logger.info(f"Чтение файла {file_path} ...")
    df = pd.DataFrame()
    try:
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            exit_with_error(logger, f"Неподдерживаемый формат файла: {file_path}")
        if not df.empty:
            df = rename_dataframe(df)
            df = df.drop_duplicates(subset=[id_field_name], keep="last")
            df = dedup_by_question_any(df)
        else:
            raise Exception
    except Exception as e:
        exit_with_error(logger, f"Ошибка при загрузке файла {file_path}: {e!r}")

    return prepare_dataframe(df, logger)


async def load_collection_and_index(
    settings: BaseSettings,
    logger: logging.Logger,
    collection_name: str = "kb_default",
) -> None:
    from app.infrastructure.adapters.open_search import OpenSearchAdapter
    from sentence_transformers import SentenceTransformer
    from app.infrastructure.storages.milvus import MilvusDatabase

    milvus = None
    model = None

    try:
        # --- Milvus ---
        logger.info("Подключение к Milvus DB ...")
        try:
            milvus = MilvusDatabase(settings=settings.milvus, logger=logger)
        except Exception as e:
            exit_with_error(logger, f"Произошла ошибка при подключении к Milvus DB: {e!r}")

        # --- Model ---
        logger.info(f"Загрузка модели {settings.milvus.model_name.split('/')[-1]} ...")
        try:
            model = SentenceTransformer(settings.milvus.model_name)
        except Exception as e:
            exit_with_error(logger, f"Произошла ошибка при загрузке модели: {e!r}")
        else:
            logger.info("Модель успешно загружена")

        logger.info(
            "Загрузка коллекции Milvus DB "
            f"({'с пересозданием коллекции' if settings.milvus.recreate_collection else 'без пересоздания коллекции'}) ..."
        )

        collection_exists = await milvus.collection_ready(collection_name)
        collection_empty = (not collection_exists) or (not await milvus.collection_not_empty(collection_name))

        recreate_milvus_collection = (
            settings.milvus.recreate_collection or (not collection_exists) or collection_empty
        )

        metadata = None  # понадобится для OpenSearch
        if recreate_milvus_collection or settings.opensearch.recreate_index:
            documents, metadata, df = prepare_data_from_file(settings.app.collection_file_path, 'ext_id', logger)

            if recreate_milvus_collection:
                if collection_exists:
                    await milvus.safe_delete_collection(collection_name)

                await milvus.initialize_collection(
                    collection_name=collection_name,
                    model=model,
                    documents=documents,
                    metadata=metadata,
                )

        await milvus.preload_collections()
        if not await milvus.collection_not_empty(collection_name):
            exit_with_error(logger, "❌ Milvus коллекция пуста или не создана!")

        # --- OpenSearch ---
        os_adapter = OpenSearchAdapter(settings=settings, logger=logger)

        if settings.opensearch.recreate_index:
            logger.info("Построение индекса OpenSearch ...")
            try:
                os_adapter.build_index(data=metadata or [])
            except Exception as e:
                exit_with_error(logger, f"Произошла ошибка при загрузке индекса OpenSearch: {e!r}")

        os_adapter.ensure_index_not_empty()

    finally:
        # закрываем Milvus даже при SystemExit/Exception
        if milvus is not None:
            try:
                await milvus.close()
            except Exception:
                logger.exception("Не удалось корректно закрыть Milvus")

        # освобождаем ресурсы модели (GPU/память) всегда
        cleanup_resources(logger, model)



async def pre_launch() -> None:
    """Скрипт, запускающийся перед приложением"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--logtype", help="Тип логирования", choices=["app", "celery"], default="app"
    )
    parser.add_argument(
        "--load", help="Загрузка коллекции в Milvus DB", action="store_true"
    )
    args = parser.parse_args()

    if args.local:
        os.environ["MILVUS_HOST"] = "localhost"
        os.environ["MILVUS_MODEL_NAME"] = (
            f"{os.environ['APP_MODELSTORE_HOST_PATH']}/{os.environ['MILVUS_MODEL_NAME']}"
        )
        os.environ["RERANKER_MODEL_NAME"] = (
            f"{os.environ['APP_MODELSTORE_HOST_PATH']}/{os.environ['RERANKER_MODEL_NAME']}"
        )
        os.environ["OS_HOST"] = "localhost"

    from app.settings.config import Settings

    settings = Settings()

    logger = setup_logger(args.logtype)

    logger.info("Запуск pre_launch ...")
    if args.load:
        logger.info("Загрузка данных в Milvus DB...")
        try:
            await load_collection_and_index(
                settings, logger, settings.milvus.collection_name
            )
        except Exception as e:
            exit_with_error(
                logger,
                "Произошла неожиданная ошибка в работе pre_launch -> load_collection_and_index: "
                + f"({type(e)}): {traceback.format_exc()}",
            )

    logger.info("Завершение pre_launch (0)")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(pre_launch())
