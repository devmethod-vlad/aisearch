import argparse
import asyncio
import datetime
import logging
import os
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from app.infrastructure.utils.prepare_dataframe import (
    combine_validated_sources,
    dedup_by_question_any,
    load_files_from_directory,
    prepare_dataframe,
    rename_dataframe,
    reorder_columns_by_mapping,
    validate_dataframe,
)
from app.infrastructure.utils.universal import (
    cleanup_resources,
    exit_with_error,
    get_timezone,
)


def setup_logger() -> logging.Logger:
    """Настраивает и возвращает логгер для pre_launch скрипта."""
    prefix = "[pre_launch.py]"

    logger = logging.getLogger("celery")

    original_handlers = logger.handlers.copy()
    for handler in original_handlers:
        logger.removeHandler(handler)

    logs_path = os.getenv("CELERY_LOGS_PATH")
    if not logs_path:
        raise ValueError("Переменная окружения CELERY_LOGS_PATH не установлена")

    log_file = Path(logs_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    class CustomFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            ts = datetime.datetime.now(tz=get_timezone()).strftime("%Y-%m-%d %H:%M:%S")
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


async def export_data(
    df_final: pd.DataFrame,
    base_output_dir: Path,
    mapping_config_path: str | Path,
    id_column: str,
    logger: logging.Logger,
    filename: str = "prelaunch",
) -> None:
    """Экспортирует обработанные данные в xlsx и parquet форматах."""
    try:
        df_ordered = reorder_columns_by_mapping(
            df_final, mapping_config_path, id_column
        )
    except Exception as e:
        logger.error(f"❌ Ошибка упорядочивания колонок: {e}")
        df_ordered = df_final

    timestamp = datetime.datetime.now(tz=get_timezone()).strftime("%d%m%y%H%M")
    data_dir_name = f"data_{timestamp}"
    data_dir_path = base_output_dir / data_dir_name

    data_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Создана директория: {data_dir_path}")

    # Сохраняем в XLSX
    xlsx_path = data_dir_path / f"{filename}.xlsx"
    try:
        df_ordered.to_excel(xlsx_path, index=False)
        logger.info(f"✅ Данные экспортированы в XLSX: {xlsx_path}")
        logger.info(
            f"   Размер файла: {len(df_ordered)} строк, {len(df_ordered.columns)} колонок"
        )
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения в XLSX: {e}")
        raise

    # Сохраняем в Parquet
    parquet_path = data_dir_path / f"{filename}.parquet"
    try:
        df_ordered.to_parquet(parquet_path, index=False)
        logger.info(f"✅ Данные экспортированы в Parquet: {parquet_path}")
        logger.info(
            f"   Размер файла: {len(df_ordered)} строк, {len(df_ordered.columns)} колонок"
        )
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения в Parquet: {e}")
        raise

    logger.info(f"✅ Обработанные данные экспортированы в директории: {data_dir_path}")
    files_in_dir = list(data_dir_path.iterdir())
    logger.info(f"Файлы в директории: {[f.name for f in files_in_dir]}")


async def prepare_and_load_data(
    settings: BaseSettings,
    logger: logging.Logger,
) -> tuple[list[str], list[dict], pd.DataFrame]:
    """Подготавливает данные для загрузки в базы."""
    logger.info(
        f"Загрузка данных из директории: {settings.app.collection_files_contr_dir}"
    )

    try:
        loaded_files = load_files_from_directory(
            files_dir=settings.app.collection_files_contr_dir,
        )
        logger.info(f"Найдено файлов для обработки: {len(loaded_files)}")

        validated_dfs = []
        for file_path, df in loaded_files:
            logger.info(f"Обработка файла: {file_path.name} ({len(df)} строк)")

            try:
                df_renamed = rename_dataframe(
                    df, settings.app.field_mapping_schema_path
                )
            except Exception as e:
                logger.error(f"  ↳ ❌ Ошибка переименования: {e}")
                raise

            try:
                df_validated = validate_dataframe(
                    df_renamed,
                    settings.app.field_mapping_schema_path,
                    id_column=settings.app.data_unique_id,
                )
            except Exception as e:
                logger.error(f"  ↳ ❌ Ошибка валидации: {e}")
                raise

            validated_dfs.append(df_validated)
            logger.info(f"✅ Файл {file_path.name} успешно обработан")

        logger.info(f"Все файлы обработаны: {len(validated_dfs)} шт.")

        logger.info(f"Объединение {len(validated_dfs)} DataFrame...")
        try:
            combined_df = combine_validated_sources(
                validated_dfs, id_column=settings.app.data_unique_id
            )
            logger.info(f"✅ Объединенный DataFrame: {len(combined_df)} строк")
        except Exception as e:
            logger.error(f"❌ Ошибка объединения: {e}")
            raise

        logger.info("Дедупликация по вопросу...")
        try:
            combined_df = dedup_by_question_any(combined_df)
            logger.info(f"✅ После дедупликации: {len(combined_df)} строк")
        except Exception as e:
            logger.error(f"❌ Ошибка дедупликации: {e}")
            raise

        logger.info("Подготовка финального датафрейма...")
        try:
            documents, metadata, df_final = prepare_dataframe(
                combined_df, id_column=settings.app.data_unique_id
            )
            logger.info(
                f"✅ Данные подготовлены: "
                f"{len(documents)} документов, {len(df_final)} записей"
            )
            return documents, metadata, df_final
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки данных: {e}")
            raise

    except Exception as e:
        exit_with_error(
            logger, f"Ошибка при обработке данных: {type(e)}): {traceback.format_exc()}"
        )


async def should_recreate_milvus(
    milvus_db,  # noqa: ANN001
    collection_name: str,
    settings: BaseSettings,
    logger: logging.Logger,
) -> bool:
    """Определяет, нужно ли пересоздавать коллекцию Milvus."""
    if settings.app.recreate_data:
        logger.info("Флаг recreate_data включен - Milvus будет пересоздан")
        return True

    collection_exists = await milvus_db.collection_ready(collection_name)

    if not collection_exists:
        logger.info("Коллекция Milvus не существует - будет создана")
        return True

    collection_empty = not await milvus_db.collection_not_empty(collection_name)
    if collection_empty:
        logger.info("Коллекция Milvus пуста - будет пересоздана")
        return True

    logger.info("Коллекция Milvus существует и не пуста - пересоздание не требуется")
    return False


async def should_recreate_opensearch(
    os_adapter,  # noqa: ANN001
    settings: BaseSettings,
    logger: logging.Logger,
) -> bool:
    """Определяет, нужно ли пересоздавать индекс OpenSearch."""
    if settings.app.recreate_data:
        logger.info("Флаг recreate_data включен - OpenSearch будет пересоздан")
        return True

    index_exists = await os_adapter.index_exists()

    if not index_exists:
        logger.info("Индекс OpenSearch не существует - будет создан")
        return True

    # Проверяем, не пустой ли индекс
    try:
        count = await os_adapter.count()
        if count == 0:
            logger.info("Индекс OpenSearch пуст - будет пересоздан")
            return True
    except Exception as e:
        logger.warning(f"Не удалось проверить количество документов в OpenSearch: {e}")
        # В случае ошибки считаем, что нужно пересоздать
        return True

    logger.info("Индекс OpenSearch существует и не пуст - пересоздание не требуется")
    return False


async def check_milvus_status(
    milvus_db,  # noqa: ANN001
    collection_name: str,
    logger: logging.Logger,
) -> None:
    """Проверяет статус коллекции Milvus."""
    if not await milvus_db.collection_not_empty(collection_name):
        exit_with_error(logger, "❌ Milvus коллекция пуста или не создана!")
    else:
        logger.info("✅ Milvus коллекция готова к работе")


async def check_opensearch_status(
    os_adapter,  # noqa: ANN001
    logger: logging.Logger,
) -> None:
    """Проверяет статус индекса OpenSearch."""
    try:
        count = await os_adapter.count()
        if count == 0:
            exit_with_error(logger, "Индекс OpenSearch пуст")
        logger.info(f"✅ OpenSearch индекс готов к работе (документов: {count})")
    except Exception as e:
        exit_with_error(
            logger,
            f"Проблема с индексом OpenSearch: {type(e)}): {traceback.format_exc()}",
        )


async def load_model(  # noqa: ANN201
    model_name: str,
    logger: logging.Logger,
):
    """Загружает модель для Milvus."""
    logger.info(f"Загрузка модели {model_name.split('/')[-1]} ...")
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_name, local_files_only=True)
        logger.info("Модель успешно загружена")
        return model
    except Exception as e:
        exit_with_error(
            logger,
            f"Произошла ошибка при загрузке модели: {type(e)}): {traceback.format_exc()}",
        )


async def recreate_milvus_collection(
    milvus_db,  # noqa: ANN001
    collection_name: str,
    model,  # noqa: ANN001
    documents: list,
    metadata: list,
    logger: logging.Logger,
) -> None:
    """Пересоздает коллекцию Milvus с новыми данными."""
    # Удаляем существующую коллекцию, если она есть
    collection_exists = await milvus_db.collection_ready(collection_name)
    if collection_exists:
        logger.info("Удаление существующей коллекции Milvus...")
        try:
            await milvus_db.delete_collection(collection_name)
            logger.info("✅ Коллекция Milvus удалена")
        except Exception as e:
            logger.error(f"❌ Ошибка удаления коллекции Milvus: {e}")
            raise

    # Создаем новую коллекцию
    logger.info("Создание новой коллекции Milvus...")
    try:
        await milvus_db.initialize_collection(
            collection_name=collection_name,
            model=model,
            documents=documents,
            metadata=metadata,
        )
        logger.info("✅ Коллекция Milvus создана и заполнена")
    except Exception as e:
        logger.error(f"❌ Ошибка создания коллекции Milvus: {e}")
        raise


async def recreate_opensearch_index(
    os_adapter,  # noqa: ANN001
    metadata: list,
    logger: logging.Logger,
) -> None:
    """Пересоздает индекс OpenSearch с новыми данными."""
    index_name = os_adapter.os_schema.index_name

    # Удаляем существующий индекс, если он есть
    if await os_adapter.index_exists():
        logger.info(f"Удаление индекса {index_name}...")
        try:
            await os_adapter.delete_index()
            logger.info(f"✅ Индекс {index_name} удален")
        except Exception as e:
            logger.error(f"❌ Ошибка удаления индекса OpenSearch: {e}")
            raise

    # Создаем новый индекс
    logger.info(f"Создание индекса {index_name}...")
    try:
        await os_adapter.create_index()
        logger.info(f"✅ Индекс {index_name} создан")
    except Exception as e:
        logger.error(f"❌ Ошибка создания индекса OpenSearch: {e}")
        raise

    # Загружаем данные
    logger.info(f"Загрузка {len(metadata)} документов в OpenSearch...")
    try:
        await os_adapter.build_index_with_data(data=metadata)
        logger.info("✅ Данные загружены в OpenSearch")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки данных в OpenSearch: {e}")
        raise


async def export_only(
    settings: BaseSettings,
    logger: logging.Logger,
) -> None:
    """Рабочий процесс только для экспорта данных."""
    logger.info("Режим только экспорта (--export-only)")
    documents, metadata, df_final = await prepare_and_load_data(settings, logger)

    logger.info("Экспорт данных в файлы...")
    try:
        base_output_dir = Path(settings.app.prelaunch_data_contr_dir)
        await export_data(
            df_final,
            base_output_dir,
            settings.app.field_mapping_schema_path,
            settings.app.data_unique_id,
            logger,
        )
        logger.info("✅ Экспорт данных завершен")
    except Exception as e:
        logger.error(f"❌ Ошибка экспорта данных: {e}")
        raise


async def main(
    settings: BaseSettings,
    logger: logging.Logger,
) -> None:
    """Основной рабочий процесс: загрузка данных в базы."""
    from app.infrastructure.adapters.open_search import OpenSearchAdapter
    from app.infrastructure.storages.milvus import MilvusDatabase

    milvus_db = None
    model = None
    os_adapter = None
    documents = None
    metadata = None
    df_final = None

    try:
        logger.info("Подключение к Milvus DB ...")
        try:
            milvus_db = MilvusDatabase(settings=settings.milvus, logger=logger)
        except Exception as e:
            exit_with_error(
                logger,
                f"Произошла ошибка при подключении к Milvus DB: {type(e)}): {traceback.format_exc()}",
            )

        logger.info("Подключение к OpenSearch ...")
        try:
            os_adapter = OpenSearchAdapter(settings=settings, logger=logger)
        except Exception as e:
            exit_with_error(
                logger,
                f"Произошла ошибка при подключении к OpenSearch: {type(e)}): {traceback.format_exc()}",
            )

        # Определяем, нужно ли пересоздавать базы данных
        recreate_milvus = await should_recreate_milvus(
            milvus_db, settings.milvus.collection_name, settings, logger
        )
        recreate_opensearch = await should_recreate_opensearch(
            os_adapter, settings, logger
        )

        # Если не нужно пересоздавать базы данных, просто проверяем их и выходим
        if not recreate_milvus and not recreate_opensearch:
            logger.info(
                "Базы данных уже существуют и не пусты, обновление не требуется"
            )
            return

        # Если нужно пересоздать базы данных, готовим данные
        documents, metadata, df_final = await prepare_and_load_data(settings, logger)

        # Экспорт данных, если включен флаг generate_prelaunch_data
        if settings.app.generate_prelaunch_data:
            logger.info("Экспорт данных в файлы (generate_prelaunch_data=True)...")
            try:
                base_output_dir = Path(settings.app.prelaunch_data_contr_dir)
                await export_data(
                    df_final,
                    base_output_dir,
                    settings.app.field_mapping_schema_path,
                    settings.app.data_unique_id,
                    logger,
                )
                logger.info("✅ Экспорт данных завершен")
            except Exception as e:
                logger.error(f"❌ Ошибка экспорта данных: {e}")
                raise

        # Загружаем в Milvus, если нужно пересоздать Milvus
        if recreate_milvus:
            model = await load_model(settings.milvus.model_name, logger)
            await recreate_milvus_collection(
                milvus_db,
                settings.milvus.collection_name,
                model,
                documents,
                metadata,
                logger,
            )

        # Предзагрузка коллекций Milvus
        await milvus_db.preload_collections()

        # Загрузка в OpenSearch, если нужно
        if recreate_opensearch:
            await recreate_opensearch_index(os_adapter, metadata, logger)

    finally:
        # Проверяем статус баз данных
        if milvus_db is not None:
            try:
                await check_milvus_status(
                    milvus_db, settings.milvus.collection_name, logger
                )
            except Exception:
                logger.exception("Ошибка при проверке статуса Milvus")

        if os_adapter is not None:
            try:
                await check_opensearch_status(os_adapter, logger)
            except Exception:
                logger.exception("Ошибка при проверке статуса OpenSearch")

        # Закрытие соединений и освобождение ресурсов
        if milvus_db is not None:
            try:
                await milvus_db.close()
                logger.info("Milvus соединение закрыто")
            except Exception:
                logger.exception("Не удалось корректно закрыть Milvus")

        if os_adapter is not None:
            try:
                await os_adapter.close()
                logger.info("OpenSearch соединение закрыто")
            except Exception:
                logger.exception("Не удалось корректно закрыть OpenSearch")

        cleanup_resources(logger, model)
        if model is not None:
            logger.info("Ресурсы модели освобождены")


async def pre_launch() -> None:
    """Скрипт, запускающийся перед приложением"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-only",
        help="Только экспорт данных в файлы, без загрузки в базы данных",
        action="store_true",
    )
    args = parser.parse_args()

    from app.settings.config import Settings

    settings = Settings()

    logger = setup_logger()

    logger.info("Запуск pre_launch ...")

    try:
        if args.export_only:
            await export_only(settings, logger)
        else:
            await main(settings, logger)
    except Exception as e:
        exit_with_error(
            logger,
            "Произошла неожиданная ошибка в работе pre_launch: "
            + f"({type(e)}): {traceback.format_exc()}",
        )

    logger.info("Завершение pre_launch (0)")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(pre_launch())
