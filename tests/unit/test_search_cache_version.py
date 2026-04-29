from unittest.mock import AsyncMock

import pytest

from app.infrastructure.utils.search_cache_version import (
    build_search_cache_key,
    build_search_data_version_key,
    bump_search_data_version,
    get_or_create_search_data_version,
)


def test_build_search_data_version_key() -> None:
    assert (
        build_search_data_version_key("kb_default", "search_index")
        == "hyb:meta:data_version:kb_default:search_index"
    )


def test_build_search_cache_key() -> None:
    key = build_search_cache_key(
        collection_name="kb_default",
        index_name="search_index",
        data_version="dv-1",
        hybrid_version="v1",
        query_hash="qhash",
        top_k=5,
        presearch_key_part="1:ext_id:phash",
        filters_key_part="role_tokens=admin",
    )
    assert (
        key
        == "hyb:cache:kb_default:search_index:dv-1:v1:qhash:5:1:ext_id:phash:role_tokens=admin"
    )


@pytest.mark.asyncio
async def test_get_or_create_initial_version_when_missing() -> None:
    redis = AsyncMock()
    redis.get = AsyncMock(side_effect=[None, "created-v1"])
    redis.set = AsyncMock(return_value=True)

    version = await get_or_create_search_data_version(
        redis,
        collection_name="kb_default",
        index_name="search_index",
    )

    assert version == "created-v1"
    redis.set.assert_awaited_once()
    assert redis.set.await_args.kwargs["not_exist"] is True


@pytest.mark.asyncio
async def test_get_or_create_keeps_existing_value() -> None:
    redis = AsyncMock()
    redis.get = AsyncMock(return_value="existing-v1")
    redis.set = AsyncMock()

    version = await get_or_create_search_data_version(
        redis,
        collection_name="kb_default",
        index_name="search_index",
    )

    assert version == "existing-v1"
    redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_bump_search_data_version_overwrites_value() -> None:
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)

    version = await bump_search_data_version(
        redis,
        collection_name="kb_default",
        index_name="search_index",
        reason="test",
    )

    assert isinstance(version, str)
    assert version
    redis.set.assert_awaited_once()
    assert redis.set.await_args.args[0] == "hyb:meta:data_version:kb_default:search_index"
    assert redis.set.await_args.args[1] == version
