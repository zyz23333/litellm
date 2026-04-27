from unittest.mock import MagicMock

import litellm
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.videos import avideo_delete, video_content, video_delete, video_status


def test_video_delete_is_exported_from_litellm_package():
    assert callable(litellm.video_delete)
    assert callable(litellm.avideo_delete)
    assert callable(video_delete)
    assert callable(avideo_delete)


def test_video_delete_routes_to_http_handler(monkeypatch):
    mock_logging = MagicMock()
    mock_logging.update_from_kwargs = MagicMock()
    mock_handler = MagicMock()
    mock_handler.video_delete_handler.return_value = VideoObject(
        id="cgt-delete-123",
        object="video",
        status="deleted",
    )
    monkeypatch.setattr("litellm.videos.main.base_llm_http_handler", mock_handler)

    result = litellm.video_delete(
        video_id="video:volcengine:doubao-seedance-2-0-260128:cgt-delete-123",
        litellm_logging_obj=mock_logging,
    )

    assert result.status == "deleted"
    mock_handler.video_delete_handler.assert_called_once()
    kwargs = mock_handler.video_delete_handler.call_args.kwargs
    assert kwargs["custom_llm_provider"] == "volcengine"
    assert isinstance(kwargs["litellm_params"], GenericLiteLLMParams)


def test_video_status_routes_colon_encoded_id_to_provider(monkeypatch):
    mock_logging = MagicMock()
    mock_logging.update_from_kwargs = MagicMock()
    mock_handler = MagicMock()
    mock_handler.video_status_handler.return_value = VideoObject(
        id="cgt-status-123",
        object="video",
        status="queued",
    )
    monkeypatch.setattr("litellm.videos.main.base_llm_http_handler", mock_handler)

    result = video_status(
        video_id="video:volcengine:doubao-seedance-2-0-260128:cgt-status-123",
        litellm_logging_obj=mock_logging,
    )

    assert result.status == "queued"
    mock_handler.video_status_handler.assert_called_once()
    kwargs = mock_handler.video_status_handler.call_args.kwargs
    assert kwargs["custom_llm_provider"] == "volcengine"
    assert isinstance(kwargs["litellm_params"], GenericLiteLLMParams)


def test_video_content_routes_colon_encoded_id_to_provider(monkeypatch):
    mock_logging = MagicMock()
    mock_logging.update_from_kwargs = MagicMock()
    mock_handler = MagicMock()
    mock_handler.video_content_handler.return_value = b"video-bytes"
    monkeypatch.setattr("litellm.videos.main.base_llm_http_handler", mock_handler)

    result = video_content(
        video_id="video:volcengine:doubao-seedance-2-0-260128:cgt-content-123",
        litellm_logging_obj=mock_logging,
    )

    assert result == b"video-bytes"
    mock_handler.video_content_handler.assert_called_once()
    kwargs = mock_handler.video_content_handler.call_args.kwargs
    assert kwargs["custom_llm_provider"] == "volcengine"
    assert isinstance(kwargs["litellm_params"], GenericLiteLLMParams)
