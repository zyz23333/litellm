from unittest.mock import MagicMock

import pytest
from starlette.requests import Request
from starlette.responses import Response

import litellm.proxy.proxy_server as proxy_server
import litellm.proxy.video_endpoints.endpoints as video_endpoints
from litellm.proxy.video_endpoints.endpoints import router


def test_video_delete_endpoint_exists():
    delete_routes = {
        route.path
        for route in router.routes
        if hasattr(route, "methods") and "DELETE" in route.methods
    }

    assert "/v1/videos/{video_id}" in delete_routes
    assert "/videos/{video_id}" in delete_routes


@pytest.mark.asyncio
async def test_should_pass_variant_query_param_to_video_content_processing(
    monkeypatch,
):
    captured = {}

    class FakeProcessor:
        def __init__(self, data):
            captured["data"] = data

        async def base_process_llm_request(self, **kwargs):
            captured["route_type"] = kwargs["route_type"]
            return b"last-frame-bytes"

    monkeypatch.setattr(video_endpoints, "ProxyBaseLLMRequestProcessing", FakeProcessor)
    monkeypatch.setattr(proxy_server, "general_settings", {}, raising=False)
    monkeypatch.setattr(proxy_server, "llm_router", None, raising=False)
    monkeypatch.setattr(proxy_server, "proxy_config", MagicMock(), raising=False)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock(), raising=False)
    monkeypatch.setattr(proxy_server, "select_data_generator", None, raising=False)
    monkeypatch.setattr(proxy_server, "user_api_base", None, raising=False)
    monkeypatch.setattr(proxy_server, "user_max_tokens", None, raising=False)
    monkeypatch.setattr(proxy_server, "user_model", None, raising=False)
    monkeypatch.setattr(proxy_server, "user_request_timeout", None, raising=False)
    monkeypatch.setattr(proxy_server, "user_temperature", None, raising=False)
    monkeypatch.setattr(proxy_server, "version", "test", raising=False)

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/videos/cgt-content-123/content",
            "query_string": b"variant=last_frame&custom_llm_provider=volcengine",
            "headers": [],
        }
    )

    response = await video_endpoints.video_content(
        video_id="cgt-content-123",
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=MagicMock(),
    )

    assert response.body == b"last-frame-bytes"
    assert response.headers["content-type"] == "image/jpeg"
    assert (
        response.headers["content-disposition"]
        == "attachment; filename=video_cgt-content-123_last_frame.jpg"
    )
    assert captured["route_type"] == "avideo_content"
    assert captured["data"]["video_id"] == "cgt-content-123"
    assert captured["data"]["variant"] == "last_frame"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("variant", "expected_media_type", "expected_filename"),
    [
        ("thumbnail", "image/jpeg", "video_cgt-content-123_thumbnail.jpg"),
        ("spritesheet", "image/jpeg", "video_cgt-content-123_spritesheet.jpg"),
    ],
)
async def test_should_set_image_response_metadata_for_image_content_variants(
    monkeypatch,
    variant,
    expected_media_type,
    expected_filename,
):
    class FakeProcessor:
        def __init__(self, data):
            pass

        async def base_process_llm_request(self, **kwargs):
            return b"image-bytes"

    monkeypatch.setattr(video_endpoints, "ProxyBaseLLMRequestProcessing", FakeProcessor)
    monkeypatch.setattr(proxy_server, "general_settings", {}, raising=False)
    monkeypatch.setattr(proxy_server, "llm_router", None, raising=False)
    monkeypatch.setattr(proxy_server, "proxy_config", MagicMock(), raising=False)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock(), raising=False)
    monkeypatch.setattr(proxy_server, "select_data_generator", None, raising=False)
    monkeypatch.setattr(proxy_server, "user_api_base", None, raising=False)
    monkeypatch.setattr(proxy_server, "user_max_tokens", None, raising=False)
    monkeypatch.setattr(proxy_server, "user_model", None, raising=False)
    monkeypatch.setattr(proxy_server, "user_request_timeout", None, raising=False)
    monkeypatch.setattr(proxy_server, "user_temperature", None, raising=False)
    monkeypatch.setattr(proxy_server, "version", "test", raising=False)

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/videos/cgt-content-123/content",
            "query_string": (f"variant={variant}&custom_llm_provider=openai").encode(),
            "headers": [],
        }
    )

    response = await video_endpoints.video_content(
        video_id="cgt-content-123",
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=MagicMock(),
    )

    assert response.body == b"image-bytes"
    assert response.headers["content-type"] == expected_media_type
    assert (
        response.headers["content-disposition"]
        == f"attachment; filename={expected_filename}"
    )
