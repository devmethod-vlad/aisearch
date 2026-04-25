from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

import app.infrastructure.adapters.edu as edu_module
from app.infrastructure.adapters.edu import EduAdapter


def _build_settings() -> SimpleNamespace:
    extract = SimpleNamespace(
        edu_emias_url="http://confluence.local",
        edu_emias_token="token",
        edu_emias_attachments_page_id="42",
        vio_base_file_name="vio.xlsx",
        knowledge_base_file_name="kb.xlsx",
        base_harvester_api_url="http://harvester.local",
        kb_harvester_suffix="/kb",
        vio_harvester_suffix="/vio",
        edu_timeout=10,
        deduplicated_excel_max_retries=2,
        deduplicated_excel_retry_backoff_base_seconds=0,
        deduplicated_excel_retry_backoff_max_seconds=0,
    )
    return SimpleNamespace(extract_edu=extract)


@pytest.mark.asyncio
async def test_upload_or_update_creates_attachment_when_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))

        if request.method == "GET" and "child/attachment" in str(request.url):
            return httpx.Response(200, json={"results": [], "size": 0, "_links": {}})

        if request.method == "POST" and str(request.url).endswith("/child/attachment"):
            return httpx.Response(200, json={"results": [{"id": "att-1"}]})

        if request.method == "GET" and str(request.url).endswith("/content/att-1/version?start=0&limit=100"):
            return httpx.Response(
                200,
                json={
                    "results": [{"number": 1}],
                    "size": 1,
                    "_links": {},
                },
            )

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return httpx.AsyncClient(*args, **kwargs)

    monkeypatch.setattr(edu_module.httpx, "AsyncClient", client_factory)

    adapter = EduAdapter(settings=_build_settings(), logger=MagicMock())

    await adapter.upload_or_update_attachment_to_edu(
        filename="statistic.xlsx",
        content=b"xlsx",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        keep_last_versions=3,
    )

    assert any(
        method == "POST" and url.endswith("/child/attachment")
        for method, url in calls
    )


@pytest.mark.asyncio
async def test_upload_or_update_updates_existing_and_prunes_old_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append((request.method, url))

        if request.method == "GET" and "filename=statistic.xlsx" in url:
            return httpx.Response(
                200,
                json={
                    "results": [{"id": "att-2", "title": "statistic.xlsx"}],
                    "size": 1,
                    "_links": {},
                },
            )

        if request.method == "POST" and url.endswith("/child/attachment/att-2/data"):
            return httpx.Response(200, json={"results": [{"id": "att-2"}]})

        if request.method == "GET" and url.endswith("/content/att-2/version?start=0&limit=100"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"number": 1},
                        {"number": 2},
                        {"number": 3},
                        {"number": 4},
                    ],
                    "size": 4,
                    "_links": {},
                },
            )

        if request.method == "DELETE" and url.endswith("/content/att-2/version/1"):
            return httpx.Response(204)

        if request.method == "DELETE" and url.endswith("/content/att-2/version/2"):
            return httpx.Response(204)

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return httpx.AsyncClient(*args, **kwargs)

    monkeypatch.setattr(edu_module.httpx, "AsyncClient", client_factory)

    adapter = EduAdapter(settings=_build_settings(), logger=MagicMock())

    await adapter.upload_or_update_attachment_to_edu(
        filename="statistic.xlsx",
        content=b"xlsx",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        keep_last_versions=2,
    )

    assert any(
        method == "POST" and url.endswith("/child/attachment/att-2/data")
        for method, url in calls
    )
    delete_urls = [url for method, url in calls if method == "DELETE"]
    assert any(url.endswith("/version/1") for url in delete_urls)
    assert any(url.endswith("/version/2") for url in delete_urls)
    assert all(not url.endswith("/version/4") for url in delete_urls)
