import json
from typing import Any, Dict

import httpx
import pytest

import litellm
from litellm.llms.volcengine.videos.transformation import VolcEngineVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider
from litellm.utils import ProviderConfigManager


def make_response(payload: Dict[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "https://ark.cn-beijing.volces.com/api/v3/test"),
    )


def test_provider_registration_resolves_volcengine_video_config():
    config = ProviderConfigManager.get_provider_video_config(
        model="doubao-seedance-2-0-260128",
        provider=litellm.LlmProviders.VOLCENGINE,
    )

    assert isinstance(config, VolcEngineVideoConfig)


def test_validate_environment_uses_ark_api_key(monkeypatch):
    monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    config = VolcEngineVideoConfig()

    headers = config.validate_environment(
        headers={},
        model="doubao-seedance-2-0-260128",
        litellm_params=GenericLiteLLMParams(),
    )

    assert headers["Authorization"] == "Bearer ark-test-key"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_uses_volcengine_fallback_key(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.setenv("VOLCENGINE_API_KEY", "volcengine-test-key")
    config = VolcEngineVideoConfig()

    headers = config.validate_environment(
        headers={},
        model="doubao-seedance-2-0-260128",
        litellm_params=GenericLiteLLMParams(),
    )

    assert headers["Authorization"] == "Bearer volcengine-test-key"


def test_get_complete_url_defaults_to_ark_data_plane_base():
    config = VolcEngineVideoConfig()

    assert (
        config.get_complete_url(
            model="doubao-seedance-2-0-260128",
            api_base=None,
            litellm_params={},
        )
        == "https://ark.cn-beijing.volces.com/api/v3"
    )


def test_get_complete_url_preserves_custom_api_base_without_trailing_slash():
    config = VolcEngineVideoConfig()

    assert (
        config.get_complete_url(
            model="doubao-seedance-2-0-260128",
            api_base="https://example.test/api/v3/",
            litellm_params={},
        )
        == "https://example.test/api/v3"
    )


def test_transform_create_request_builds_text_only_payload():
    config = VolcEngineVideoConfig()

    data, files, url = config.transform_video_create_request(
        model="doubao-seedance-2-0-260128",
        prompt="A cat yawning at the camera",
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        video_create_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    assert files == []
    assert data == {
        "model": "doubao-seedance-2-0-260128",
        "content": [{"type": "text", "text": "A cat yawning at the camera"}],
    }


def test_transform_create_request_builds_prompt_plus_image_payload():
    config = VolcEngineVideoConfig()

    data, files, _ = config.transform_video_create_request(
        model="doubao-seedance-2-0-260128",
        prompt="Make this image cinematic",
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        video_create_optional_request_params={
            "input_reference": "https://example.test/image.png",
            "duration": 5,
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert files == []
    assert data["content"] == [
        {"type": "text", "text": "Make this image cinematic"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.test/image.png"},
        },
    ]
    assert data["duration"] == 5
    assert "input_reference" not in data


def test_transform_create_request_prefers_ark_native_content():
    config = VolcEngineVideoConfig()
    native_content = [
        {
            "type": "image_url",
            "role": "first_frame",
            "image_url": {"url": "asset://first"},
        },
        {
            "type": "image_url",
            "role": "last_frame",
            "image_url": {"url": "asset://last"},
        },
    ]

    data, _, _ = config.transform_video_create_request(
        model="doubao-seedance-2-0-260128",
        prompt="",
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        video_create_optional_request_params={
            "content": native_content,
            "ratio": "16:9",
            "resolution": "720p",
            "duration": 5,
            "seed": 11,
            "watermark": True,
            "extra_headers": {"x-ignore": "not-body"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert data["content"] == native_content
    assert data["ratio"] == "16:9"
    assert data["resolution"] == "720p"
    assert data["duration"] == 5
    assert data["seed"] == 11
    assert data["watermark"] is True
    assert "extra_headers" not in data


def test_transform_create_request_passes_native_multimodal_content_through():
    config = VolcEngineVideoConfig()
    native_content = [
        {"type": "text", "text": "Edit this with matching audio"},
        {
            "type": "image_url",
            "role": "reference_image",
            "image_url": {"url": "https://example.test/ref.png"},
        },
        {
            "type": "video_url",
            "role": "reference_video",
            "video_url": {"url": "https://example.test/ref.mp4"},
        },
        {
            "type": "audio_url",
            "role": "reference_audio",
            "audio_url": {"url": "https://example.test/ref.mp3"},
        },
        {
            "type": "draft_task",
            "draft_task": {"id": "cgt-draft-123"},
        },
    ]

    data, files, url = config.transform_video_create_request(
        model="doubao-seedance-2-0-260128",
        prompt="ignored when native content is provided",
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        video_create_optional_request_params={
            "content": native_content,
            "generate_audio": True,
            "return_last_frame": True,
            "tools": [{"type": "web_search"}],
            "safety_identifier": "hashed-user-123",
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    assert files == []
    assert data["content"] == native_content
    assert data["generate_audio"] is True
    assert data["return_last_frame"] is True
    assert data["tools"] == [{"type": "web_search"}]
    assert data["safety_identifier"] == "hashed-user-123"


def test_map_openai_params_maps_seconds_to_duration_and_extra_body_wins():
    config = VolcEngineVideoConfig()

    mapped = config.map_openai_params(
        video_create_optional_params={
            "seconds": "5",
            "extra_body": {"duration": -1, "ratio": "adaptive"},
        },
        model="doubao-seedance-2-0-260128",
        drop_params=False,
    )

    assert mapped["duration"] == -1
    assert mapped["ratio"] == "adaptive"
    assert "seconds" not in mapped
    assert "extra_body" not in mapped


def test_map_openai_params_maps_seconds_to_integer_duration():
    config = VolcEngineVideoConfig()

    mapped = config.map_openai_params(
        video_create_optional_params={
            "seconds": "5",
        },
        model="doubao-seedance-2-0-260128",
        drop_params=False,
    )

    assert mapped["duration"] == 5
    assert "seconds" not in mapped


def test_map_openai_params_maps_user_to_safety_identifier():
    config = VolcEngineVideoConfig()

    mapped = config.map_openai_params(
        video_create_optional_params={
            "user": "hashed-user-123",
        },
        model="doubao-seedance-2-0-260128",
        drop_params=False,
    )

    assert mapped["safety_identifier"] == "hashed-user-123"
    assert "user" not in mapped


@pytest.mark.parametrize(
    ("size", "expected_resolution", "expected_ratio"),
    [
        ("1280x720", "720p", "16:9"),
        ("720x1280", "720p", "9:16"),
        ("1920x1080", "1080p", "16:9"),
        ("1080x1920", "1080p", "9:16"),
        ("960x960", None, "1:1"),
    ],
)
def test_map_openai_params_maps_size_to_ark_resolution_and_ratio(
    size, expected_resolution, expected_ratio
):
    config = VolcEngineVideoConfig()

    mapped = config.map_openai_params(
        video_create_optional_params={
            "size": size,
        },
        model="doubao-seedance-2-0-260128",
        drop_params=False,
    )

    if expected_resolution is not None:
        assert mapped["resolution"] == expected_resolution
    else:
        assert "resolution" not in mapped
    assert mapped["ratio"] == expected_ratio
    assert "size" not in mapped


def test_map_openai_params_ignores_invalid_size():
    config = VolcEngineVideoConfig()

    mapped = config.map_openai_params(
        video_create_optional_params={
            "size": "0x0",
        },
        model="doubao-seedance-2-0-260128",
        drop_params=False,
    )

    assert mapped == {}


def test_map_openai_params_does_not_override_explicit_ark_size_fields():
    config = VolcEngineVideoConfig()

    mapped = config.map_openai_params(
        video_create_optional_params={
            "size": "1280x720",
            "extra_body": {
                "resolution": "480p",
                "ratio": "adaptive",
            },
        },
        model="doubao-seedance-2-0-260128",
        drop_params=False,
    )

    assert mapped["resolution"] == "480p"
    assert mapped["ratio"] == "adaptive"
    assert "size" not in mapped


def test_transform_create_response_maps_id_only_response_to_queued_video():
    config = VolcEngineVideoConfig()

    video = config.transform_video_create_response(
        model="doubao-seedance-2-0-260128",
        raw_response=make_response({"id": "cgt-create-123"}),
        logging_obj=None,
        custom_llm_provider="volcengine",
        request_data={"model": "doubao-seedance-2-0-260128"},
    )

    decoded = decode_video_id_with_provider(video.id)
    assert decoded["video_id"] == "cgt-create-123"
    assert decoded["custom_llm_provider"] == "volcengine"
    assert video.object == "video"
    assert video.status == "queued"


@pytest.mark.parametrize(
    ("ark_status", "litellm_status"),
    [
        ("queued", "queued"),
        ("running", "processing"),
        ("succeeded", "completed"),
        ("failed", "failed"),
        ("expired", "failed"),
        ("cancelled", "cancelled"),
    ],
)
def test_transform_status_response_maps_statuses(ark_status, litellm_status):
    config = VolcEngineVideoConfig()

    video = config.transform_video_status_retrieve_response(
        raw_response=make_response(
            {
                "id": "cgt-status-123",
                "model": "doubao-seedance-2-0-260128",
                "status": ark_status,
                "created_at": 1710000000,
                "updated_at": 1710000050,
                "content": {
                    "video_url": "https://example.test/video.mp4",
                    "last_frame_url": "https://example.test/last.png",
                },
                "duration": 5,
                "resolution": "720p",
                "ratio": "16:9",
                "seed": 11,
                "framespersecond": 24,
                "generate_audio": True,
                "usage": {"completion_tokens": 100, "total_tokens": 100},
            }
        ),
        logging_obj=None,
        custom_llm_provider="volcengine",
    )

    assert video.status == litellm_status
    assert video.model == "doubao-seedance-2-0-260128"
    assert video.created_at == 1710000000
    if litellm_status in {"completed", "failed", "cancelled"}:
        assert video.completed_at == 1710000050
    assert video.seconds == "5"
    assert video.usage == {"completion_tokens": 100, "total_tokens": 100}
    assert video._hidden_params["video_url"] == "https://example.test/video.mp4"
    assert video._hidden_params["last_frame_url"] == "https://example.test/last.png"
    assert video._hidden_params["resolution"] == "720p"
    assert video._hidden_params["ratio"] == "16:9"
    assert video._hidden_params["seed"] == 11
    assert video._hidden_params["framespersecond"] == 24
    assert video._hidden_params["generate_audio"] is True


def test_transform_status_response_maps_error_payload():
    config = VolcEngineVideoConfig()

    video = config.transform_video_status_retrieve_response(
        raw_response=make_response(
            {
                "id": "cgt-failed",
                "status": "failed",
                "error": {"code": "InvalidRequest", "message": "bad request"},
                "created_at": 1710000000,
                "updated_at": 1710000060,
            }
        ),
        logging_obj=None,
        custom_llm_provider="volcengine",
    )

    assert video.status == "failed"
    assert video.error == {"code": "InvalidRequest", "message": "bad request"}
    assert video._hidden_params["ark_status"] == "failed"


def test_transform_status_request_extracts_original_video_id():
    config = VolcEngineVideoConfig()
    encoded_id = "video:volcengine:doubao-seedance-2-0-260128:cgt-status-123"

    url, data = config.transform_video_status_retrieve_request(
        video_id=encoded_id,
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert (
        url
        == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/cgt-status-123"
    )
    assert data == {}


def test_transform_content_request_uses_task_status_endpoint():
    config = VolcEngineVideoConfig()

    url, data = config.transform_video_content_request(
        video_id="cgt-status-123",
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert (
        url
        == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/cgt-status-123"
    )
    assert data == {}


def test_extract_video_url_rejects_processing_status():
    config = VolcEngineVideoConfig()

    with pytest.raises(ValueError, match="still processing"):
        config._extract_video_url_from_task({"status": "running", "id": "cgt-running"})


def test_extract_video_url_rejects_failed_status_with_ark_error():
    config = VolcEngineVideoConfig()

    with pytest.raises(ValueError, match="Video generation failed"):
        config._extract_video_url_from_task(
            {
                "status": "failed",
                "error": {"code": "Failed", "message": "model failed"},
            }
        )


def test_extract_video_url_requires_completed_output_url():
    config = VolcEngineVideoConfig()

    with pytest.raises(ValueError, match="Video URL not found"):
        config._extract_video_url_from_task({"status": "succeeded", "content": {}})


def test_extract_video_url_from_succeeded_task():
    config = VolcEngineVideoConfig()

    assert (
        config._extract_video_url_from_task(
            {
                "status": "succeeded",
                "content": {"video_url": "https://example.test/video.mp4"},
            }
        )
        == "https://example.test/video.mp4"
    )


def test_extract_output_url_from_succeeded_task_supports_last_frame_variant():
    config = VolcEngineVideoConfig()

    assert (
        config._extract_output_url_from_task(
            {
                "status": "succeeded",
                "content": {
                    "video_url": "https://example.test/video.mp4",
                    "last_frame_url": "https://example.test/last.png",
                },
            },
            variant="last_frame",
        )
        == "https://example.test/last.png"
    )


def test_extract_output_url_from_succeeded_task_rejects_missing_last_frame_variant():
    config = VolcEngineVideoConfig()

    with pytest.raises(ValueError, match="Last frame URL not found"):
        config._extract_output_url_from_task(
            {
                "status": "succeeded",
                "content": {"video_url": "https://example.test/video.mp4"},
            },
            variant="last_frame",
        )


@pytest.mark.asyncio
async def test_async_transform_content_response_downloads_asset_with_litellm_handler():
    config = VolcEngineVideoConfig()
    calls = []

    class FakeVideoResponse:
        content = b"video-bytes"
        headers = {"x-ark-message": "视频生成成功"}

        def raise_for_status(self):
            return None

    class FakeAsyncHTTPHandler:
        async def get(self, url):
            calls.append(("get", url))
            return FakeVideoResponse()

    content = await config.async_transform_video_content_response_with_variant(
        raw_response=make_response(
            {
                "status": "succeeded",
                "content": {"video_url": "https://example.test/video.mp4"},
            }
        ),
        logging_obj=None,
        async_httpx_client=FakeAsyncHTTPHandler(),
    )

    assert content == b"video-bytes"
    assert calls == [("get", "https://example.test/video.mp4")]


def test_transform_list_request_maps_limit_and_extra_query():
    config = VolcEngineVideoConfig()

    url, params = config.transform_video_list_request(
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        litellm_params=GenericLiteLLMParams(),
        headers={},
        limit=20,
        extra_query={
            "page_num": 2,
            "filter.status": "succeeded",
            "filter.task_ids": ["cgt-1", "cgt-2"],
            "filter.model": "endpoint-id",
            "filter.service_tier": "default",
        },
    )

    assert url == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    assert params == {
        "page_size": 20,
        "page_num": 2,
        "filter.status": "succeeded",
        "filter.task_ids": ["cgt-1", "cgt-2"],
        "filter.model": "endpoint-id",
        "filter.service_tier": "default",
    }


def test_transform_list_response_maps_items_envelope_and_ids():
    config = VolcEngineVideoConfig()

    result = config.transform_video_list_response(
        raw_response=make_response(
            {
                "items": [
                    {
                        "id": "cgt-1",
                        "model": "doubao-seedance-2-0-260128",
                        "status": "succeeded",
                        "created_at": 1710000000,
                        "updated_at": 1710000060,
                        "content": {"video_url": "https://example.test/1.mp4"},
                        "duration": 5,
                    },
                    {
                        "id": "cgt-2",
                        "model": "doubao-seedance-2-0-260128",
                        "status": "running",
                        "created_at": 1710000100,
                    },
                ],
                "total": 5,
            }
        ),
        logging_obj=None,
        custom_llm_provider="volcengine",
    )

    assert result["object"] == "list"
    assert result["total"] == 5
    assert result["has_more"] is True
    assert len(result["data"]) == 2
    assert result["data"][0]["status"] == "completed"
    assert result["data"][1]["status"] == "processing"
    assert result["first_id"] == result["data"][0]["id"]
    assert result["last_id"] == result["data"][1]["id"]
    assert decode_video_id_with_provider(result["data"][0]["id"])["video_id"] == "cgt-1"


def test_transform_delete_request_maps_to_ark_endpoint():
    config = VolcEngineVideoConfig()

    url, data = config.transform_video_delete_request(
        video_id="cgt-delete-123",
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert (
        url
        == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/cgt-delete-123"
    )
    assert data == {}


def test_transform_delete_response_handles_empty_body():
    config = VolcEngineVideoConfig()
    response = httpx.Response(
        status_code=204,
        content=b"",
        request=httpx.Request(
            "DELETE", "https://ark.cn-beijing.volces.com/api/v3/test"
        ),
    )

    video = config.transform_video_delete_response(
        raw_response=response,
        logging_obj=None,
    )

    assert video.object == "video"
    assert video.status == "deleted"


def test_unsupported_video_methods_raise_not_implemented():
    config = VolcEngineVideoConfig()

    with pytest.raises(NotImplementedError, match="native content"):
        config.transform_video_remix_request(
            "id", "prompt", "base", GenericLiteLLMParams(), {}
        )
    with pytest.raises(NotImplementedError, match="native content"):
        config.transform_video_edit_request(
            "prompt", "id", "base", GenericLiteLLMParams(), {}
        )
    with pytest.raises(NotImplementedError, match="native content"):
        config.transform_video_extension_request(
            "prompt", "id", "5", "base", GenericLiteLLMParams(), {}
        )
