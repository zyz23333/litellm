import math
from typing import TYPE_CHECKING, Any, Dict, NoReturn, Optional, Tuple, Union

import httpx
import litellm
from httpx._types import RequestFiles

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.volcengine.common_utils import (
    VolcEngineError,
    get_volcengine_headers,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import (
    CharacterObject,
    VideoCreateOptionalRequestParams,
    VideoObject,
)
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


ARK_VIDEO_DEFAULT_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
ARK_CREATE_PASSTHROUGH_PARAMS = {
    "content",
    "resolution",
    "ratio",
    "duration",
    "frames",
    "seed",
    "camera_fixed",
    "watermark",
    "return_last_frame",
    "service_tier",
    "execution_expires_after",
    "generate_audio",
    "draft",
    "tools",
    "safety_identifier",
    "callback_url",
}


def _build_image_content_item(input_reference: Any) -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": input_reference},
    }


def _map_openai_size_to_ark_format(size: Any) -> Dict[str, str]:
    if not isinstance(size, str) or "x" not in size:
        return {}

    try:
        width_str, height_str = size.lower().split("x", 1)
        width = int(width_str)
        height = int(height_str)
    except (TypeError, ValueError):
        return {}

    if width <= 0 or height <= 0:
        return {}

    ratio_map = {
        (16, 9): "16:9",
        (9, 16): "9:16",
        (4, 3): "4:3",
        (3, 4): "3:4",
        (1, 1): "1:1",
        (21, 9): "21:9",
    }
    gcd_value = math.gcd(width, height)
    normalized_ratio = (width // gcd_value, height // gcd_value)

    mapped: Dict[str, str] = {}
    ratio = ratio_map.get(normalized_ratio)
    if ratio is not None:
        mapped["ratio"] = ratio

    shorter_edge = min(width, height)
    if shorter_edge == 480:
        mapped["resolution"] = "480p"
    elif shorter_edge == 720:
        mapped["resolution"] = "720p"
    elif shorter_edge == 1080:
        mapped["resolution"] = "1080p"

    return mapped


def _map_openai_seconds_to_ark_duration(seconds: Any) -> Optional[int]:
    if seconds is None:
        return None

    try:
        return int(float(seconds))
    except (TypeError, ValueError):
        return None


def _extract_ark_task_id(video_id: str) -> str:
    original_video_id = extract_original_video_id(video_id)
    if (
        original_video_id == video_id
        and video_id.startswith("video:")
        and ":" in video_id
    ):
        return video_id.rsplit(":", 1)[-1]
    return original_video_id


class VolcEngineVideoConfig(BaseVideoConfig):
    def _unsupported(self, capability: str) -> NoReturn:
        raise NotImplementedError(
            f"Ark/Seedance video {capability} is not implemented as a separate OpenAI-style endpoint for Volcengine. Use video_generation with Ark-native content via extra_body for provider-specific workflows."
        )

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "extra_body",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped_params: Dict[str, Any] = {}

        if "seconds" in video_create_optional_params:
            duration = _map_openai_seconds_to_ark_duration(
                video_create_optional_params["seconds"]
            )
            if duration is not None:
                mapped_params["duration"] = duration

        if "size" in video_create_optional_params:
            mapped_size = _map_openai_size_to_ark_format(
                video_create_optional_params["size"]
            )
            mapped_params.update(mapped_size)

        if "user" in video_create_optional_params:
            mapped_params["safety_identifier"] = video_create_optional_params["user"]

        for key, value in video_create_optional_params.items():
            if key not in {"seconds", "size", "user", "extra_body"}:
                mapped_params[key] = value

        extra_body = video_create_optional_params.get("extra_body")
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is not None:
                    mapped_params[key] = value
        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params is None:
            litellm_params = GenericLiteLLMParams()
        elif isinstance(litellm_params, dict):
            litellm_params = GenericLiteLLMParams(**litellm_params)

        resolved_api_key = (
            api_key
            or litellm_params.api_key
            or litellm.api_key
            or get_secret_str("ARK_API_KEY")
            or get_secret_str("VOLCENGINE_API_KEY")
        )
        if resolved_api_key is None:
            raise ValueError(
                "Volcengine API key is required. Set ARK_API_KEY / VOLCENGINE_API_KEY or pass api_key."
            )
        return get_volcengine_headers(
            api_key=resolved_api_key,
            extra_headers=headers,
        )

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        return (api_base or ARK_VIDEO_DEFAULT_API_BASE).rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        request_data: Dict[str, Any] = {"model": model}

        native_content = video_create_optional_request_params.get("content")
        if native_content is not None:
            request_data["content"] = native_content
        else:
            content = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            input_reference = video_create_optional_request_params.get(
                "input_reference"
            )
            if input_reference is not None:
                content.append(_build_image_content_item(input_reference))
            if not content:
                raise ValueError(
                    "Volcengine video generation requires prompt, input_reference, or Ark-native content."
                )
            request_data["content"] = content

        for key in ARK_CREATE_PASSTHROUGH_PARAMS:
            if key == "content":
                continue
            if key in video_create_optional_request_params:
                request_data[key] = video_create_optional_request_params[key]

        return request_data, [], f"{api_base}/contents/generations/tasks"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        video_data: Dict[str, Any] = {
            "id": response_data.get("id", ""),
            "object": "video",
            "status": "queued",
        }
        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id,
                custom_llm_provider,
                model,
            )
        return video_obj

    def _map_ark_status(self, status: Optional[str]) -> str:
        status_map = {
            "queued": "queued",
            "running": "processing",
            "succeeded": "completed",
            "failed": "failed",
            "expired": "failed",
            "cancelled": "cancelled",
        }
        return status_map.get((status or "").lower(), "queued")

    def _is_terminal_status(self, mapped_status: str) -> bool:
        return mapped_status in {"completed", "failed", "cancelled"}

    def _hidden_params_from_task(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        content = response_data.get("content") or {}
        hidden: Dict[str, Any] = {
            "ark_status": response_data.get("status"),
        }
        for source_key, hidden_key in (
            ("video_url", "video_url"),
            ("last_frame_url", "last_frame_url"),
        ):
            if isinstance(content, dict) and content.get(source_key) is not None:
                hidden[hidden_key] = content[source_key]
        for key in (
            "seed",
            "resolution",
            "ratio",
            "frames",
            "framespersecond",
            "generate_audio",
            "service_tier",
            "execution_expires_after",
            "draft",
            "draft_task_id",
            "tools",
            "safety_identifier",
        ):
            if response_data.get(key) is not None:
                hidden[key] = response_data[key]
        return hidden

    def _video_object_from_task(
        self,
        response_data: Dict[str, Any],
        custom_llm_provider: Optional[str],
    ) -> VideoObject:
        mapped_status = self._map_ark_status(response_data.get("status"))
        video_data: Dict[str, Any] = {
            "id": response_data.get("id", ""),
            "object": "video",
            "status": mapped_status,
            "model": response_data.get("model"),
            "created_at": response_data.get("created_at"),
            "error": response_data.get("error"),
            "usage": response_data.get("usage"),
        }
        if self._is_terminal_status(mapped_status):
            video_data["completed_at"] = response_data.get("updated_at")
        if response_data.get("duration") is not None:
            video_data["seconds"] = str(response_data["duration"])

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]
        video_obj._hidden_params = self._hidden_params_from_task(response_data)
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id,
                custom_llm_provider,
                response_data.get("model"),
            )
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        original_video_id = _extract_ark_task_id(video_id)
        return f"{api_base}/contents/generations/tasks/{original_video_id}", {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        return self.transform_video_content_response_with_variant(
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        async_httpx_client: Optional[AsyncHTTPHandler] = None,
    ) -> bytes:
        return await self.async_transform_video_content_response_with_variant(
            raw_response=raw_response,
            logging_obj=logging_obj,
            async_httpx_client=async_httpx_client,
        )

    def transform_video_content_response_with_variant(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        variant: Optional[str] = None,
    ) -> bytes:
        video_url = self._extract_output_url_from_task(
            response_data=raw_response.json(),
            variant=variant,
        )
        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response_with_variant(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        variant: Optional[str] = None,
        async_httpx_client: Optional[AsyncHTTPHandler] = None,
    ) -> bytes:
        video_url = self._extract_output_url_from_task(
            response_data=raw_response.json(),
            variant=variant,
        )
        if async_httpx_client is None:
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.VOLCENGINE,
            )
        video_response = await async_httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    def _extract_video_url_from_task(self, response_data: Dict[str, Any]) -> str:
        return self._extract_output_url_from_task(response_data=response_data)

    def _extract_output_url_from_task(
        self,
        response_data: Dict[str, Any],
        variant: Optional[str] = None,
    ) -> str:
        status = (response_data.get("status") or "").lower()
        if status in {"queued", "running"}:
            raise ValueError(
                f"Video is still processing (status: {status}). Please wait and try again."
            )
        if status in {"failed", "expired", "cancelled"}:
            error = response_data.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else None
            raise ValueError(
                f"Video generation failed (status: {status}): {message or 'Unknown error'}"
            )
        content = response_data.get("content") or {}
        if not isinstance(content, dict):
            raise ValueError(
                "Video output not found in completed Volcengine task response."
            )

        if variant in {"last_frame", "thumbnail"}:
            last_frame_url = content.get("last_frame_url")
            if not last_frame_url:
                raise ValueError(
                    "Last frame URL not found in completed Volcengine task response. "
                    "Set return_last_frame=true when creating the task."
                )
            return last_frame_url

        video_url = content.get("video_url")
        if not video_url:
            raise ValueError(
                "Video URL not found in completed Volcengine task response. The task may be missing output or the temporary URL may have expired."
            )
        return video_url

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        self._unsupported("remix")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        self._unsupported("remix")

    def transform_video_create_character_request(
        self,
        name: str,
        video: Any,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, list]:
        self._unsupported("create character")

    def transform_video_create_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CharacterObject:
        self._unsupported("create character")

    def transform_video_get_character_request(
        self,
        character_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        self._unsupported("get character")

    def transform_video_get_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CharacterObject:
        self._unsupported("get character")

    def transform_video_edit_request(
        self,
        prompt: str,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        self._unsupported("edit")

    def transform_video_edit_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        self._unsupported("edit")

    def transform_video_extension_request(
        self,
        prompt: str,
        video_id: str,
        seconds: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        self._unsupported("extension")

    def transform_video_extension_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        self._unsupported("extension")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        params: Dict[str, Any] = {}
        if limit is not None:
            params["page_size"] = limit
        if extra_query:
            params.update(extra_query)
        return f"{api_base}/contents/generations/tasks", params

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        response_data = raw_response.json()
        items = response_data.get("items") or []
        mapped_items = [
            self._video_object_from_task(item, custom_llm_provider).json()
            for item in items
        ]
        total = response_data.get("total", len(mapped_items))
        result: Dict[str, Any] = {
            "object": "list",
            "data": mapped_items,
            "total": total,
            "has_more": total > len(mapped_items),
        }
        if mapped_items:
            result["first_id"] = mapped_items[0]["id"]
            result["last_id"] = mapped_items[-1]["id"]
        return result

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = _extract_ark_task_id(video_id)
        self._last_delete_video_id = original_video_id
        return f"{api_base}/contents/generations/tasks/{original_video_id}", {}

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        video_id = getattr(self, "_last_delete_video_id", "")
        return VideoObject(
            id=video_id,
            object="video",
            status="deleted",
        )  # type: ignore[arg-type]

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = _extract_ark_task_id(video_id)
        return f"{api_base}/contents/generations/tasks/{original_video_id}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        return self._video_object_from_task(
            response_data=raw_response.json(),
            custom_llm_provider=custom_llm_provider,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        typed_headers = (
            headers
            if isinstance(headers, httpx.Headers)
            else httpx.Headers(headers or {})
        )
        return VolcEngineError(
            status_code=status_code,
            message=error_message,
            headers=typed_headers,
        )
