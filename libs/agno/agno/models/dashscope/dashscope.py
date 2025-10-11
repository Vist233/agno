from dataclasses import dataclass
from os import getenv
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Type, Union

from pydantic import BaseModel

from agno.exceptions import ModelProviderError
from agno.models.message import Message
from agno.models.openai.like import OpenAILike
from agno.models.response import ModelResponse
from agno.run.agent import RunOutput


@dataclass
class DashScope(OpenAILike):
    """
    A class for interacting with Qwen models via DashScope API.

    Attributes:
        id (str): The model id. Defaults to "qwen-plus".
        name (str): The model name. Defaults to "Qwen".
        provider (str): The provider name. Defaults to "Qwen".
        api_key (Optional[str]): The DashScope API key.
        region (str): Endpoint region. Defaults to "SG" (Singapore).
        base_url (Optional[str]): The base URL. Defaults to the region-specific endpoint.
        enable_thinking (bool): Enable thinking process (DashScope native parameter). Defaults to False.
        include_thoughts (Optional[bool]): Include thinking process in response (alternative parameter). Defaults to None.
    """

    id: str = "qwen-plus"
    name: str = "Qwen"
    provider: str = "Dashscope"

    api_key: Optional[str] = getenv("DASHSCOPE_API_KEY") or getenv("QWEN_API_KEY")
    region: str = "SG"
    base_url: Optional[str] = None

    _SINGAPORE_ENDPOINT: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    _CHINA_ENDPOINT: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Thinking parameters
    enable_thinking: bool = False
    include_thoughts: Optional[bool] = None
    thinking_budget: Optional[int] = None

    # DashScope supports structured outputs
    supports_native_structured_outputs: bool = True
    supports_json_schema_outputs: bool = True

    def __post_init__(self) -> None:
        region_normalized = (self.region or "sg").lower()
        if not self.base_url:
            if region_normalized in {"cn", "china", "zh", "cn-mainland", "CN", "CHINA", "ZH", "CN-MAINLAND"}:
                self.base_url = self._CHINA_ENDPOINT
            else:
                self.base_url = self._SINGAPORE_ENDPOINT
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()

    def _ensure_json_hint(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]],
    ) -> List[Message]:
        if not response_format:
            return messages

        format_type: Optional[str] = None
        if isinstance(response_format, dict):
            format_type = response_format.get("type")
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            format_type = "json_schema"

        if format_type not in {"json_object", "json_schema"}:
            return messages

        has_json_keyword = any(isinstance(msg.content, str) and "json" in msg.content.lower() for msg in messages)
        if has_json_keyword:
            return messages

        return [Message(role="system", content="Please return in JSON format.")] + messages

    def _get_client_params(self) -> Dict[str, Any]:
        if not self.api_key:
            self.api_key = getenv("DASHSCOPE_API_KEY")
            if not self.api_key:
                raise ModelProviderError(
                    message="DASHSCOPE_API_KEY not set. Please set the DASHSCOPE_API_KEY environment variable.",
                    model_name=self.name,
                    model_id=self.id,
                )

        # Define base client params
        base_params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        # Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}

        # Add additional client params if provided
        if self.client_params:
            client_params.update(self.client_params)
        return client_params

    def invoke(
        self,
        messages: List[Message],
        assistant_message: Message,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> ModelResponse:
        messages = self._ensure_json_hint(messages, response_format)
        return super().invoke(messages, assistant_message, response_format, tools, tool_choice, run_response)

    async def ainvoke(
        self,
        messages: List[Message],
        assistant_message: Message,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> ModelResponse:
        messages = self._ensure_json_hint(messages, response_format)
        return await super().ainvoke(messages, assistant_message, response_format, tools, tool_choice, run_response)

    def invoke_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> Iterator[ModelResponse]:
        messages = self._ensure_json_hint(messages, response_format)
        return super().invoke_stream(messages, assistant_message, response_format, tools, tool_choice, run_response)

    async def ainvoke_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> AsyncIterator[ModelResponse]:
        messages = self._ensure_json_hint(messages, response_format)
        async for chunk in super().ainvoke_stream(
            messages,
            assistant_message,
            response_format,
            tools,
            tool_choice,
            run_response,
        ):
            yield chunk

    def get_request_params(
        self,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params = super().get_request_params(response_format=response_format, tools=tools, tool_choice=tool_choice)

        if self.include_thoughts is not None:
            self.enable_thinking = self.include_thoughts

        if self.enable_thinking is not None:
            params["extra_body"] = {
                "enable_thinking": self.enable_thinking,
            }

            if self.thinking_budget is not None:
                params["extra_body"]["thinking_budget"] = self.thinking_budget

        return params
