import asyncio
from logging.config import fileConfig
import traceback

import alembic_postgresql_enum
from sqlalchemy import text

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from app.infrastructure.migrations.base import Base
from app.common.logger import AISearchLogger, LoggerType

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
from app.settings.config import settings as config_settings

config = context.config

section = config.config_ini_section

config.set_main_option("sqlalchemy.url", str(config_settings.database.dsn))

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

logger = AISearchLogger(logger_type=LoggerType.CELERY)


async def create_database_if_not_exists() -> None:
    """Создает базу данных, если она не существует"""
    db_dsn = config_settings.database.dsn
    db_name = config_settings.database.db

    admin_dsn = str(db_dsn).replace(f"/{db_name}", "/postgres")

    admin_engine = create_async_engine(admin_dsn, isolation_level="AUTOCOMMIT")

    try:
        async with admin_engine.connect() as conn:
            result = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": db_name},
            )
            db_exists = result.scalar() is not None

            if not db_exists:
                logger.info(f"Creating database: {db_name}")
                await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                logger.info(f"Database {db_name} created successfully")
            else:
                logger.info(f"Database {db_name} already exists")
    except Exception as e:
        logger.error(f"Error checking/creating database ({type(e)}): {traceback.format_exc()}")
        raise
    finally:
        await admin_engine.dispose()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        process_revision_directives=add_pg_trgm_extension,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = create_async_engine(config.get_main_option("sqlalchemy.url"))

    async def run_async_migrations() -> None:
        await create_database_if_not_exists()

        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)

    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(run_async_migrations())
    else:
        loop.run_until_complete(run_async_migrations())


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        process_revision_directives=add_pg_trgm_extension,
        user_module_prefix="sa.",
    )
    with context.begin_transaction():
        context.run_migrations()


def add_pg_trgm_extension(context, revision, directives):
    """
    Хук для добавления pg_trgm только в initial миграцию
    """
    migration_script = directives[0]
    script = context.script
    if script:
        current_revisions = script.get_all_current(script.get_heads())
        is_initial = len(current_revisions) == 0
    else:
        is_initial = False

    has_trgm_extension = any(
        hasattr(op, "sqltext") and "pg_trgm" in str(op.sqltext)
        for op in migration_script.upgrade_ops.ops
    )

    if is_initial and not has_trgm_extension and migration_script.upgrade_ops.ops:
        from alembic.operations.ops import ExecuteSQLOp

        trgm_op = ExecuteSQLOp("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        migration_script.upgrade_ops.ops.insert(0, trgm_op)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
