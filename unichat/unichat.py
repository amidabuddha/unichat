from typing import Dict, Generator, List, Union, Optional

from .api_helper import _ApiHelper
from .chat_helper import _ChatHelper


class UnifiedChatApi:
    def __init__(self, api_key: str, base_url: Optional[str]=None):
        self.api_key = api_key
        self.base_url = base_url
        self._api_helper = _ApiHelper(api_key=api_key, base_url=base_url)
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, parent: 'UnifiedChatApi'):
            self._api_helper: _ApiHelper = parent._api_helper
            self.completions = self.Completions(self)

        class Completions:
            def __init__(self, parent: 'UnifiedChatApi.Chat'):
                self._api_helper: _ApiHelper = parent._api_helper
                self._chat_helper = None

            def create(
                self,
                model: str,
                messages: List[Dict[str, str]],
                temperature: str = "1.0",
                tools: Optional[List[dict]] = None,
                stream: bool = True,
                cached: Union[bool, str] = False,
                reasoning_effort: Union[bool, str] = False,
            ) -> Union[Generator, str]:
                """
                Get chat completion from various AI models.

                Args:
                    model: Name of the model to use
                    messages: List of conversation messages
                    temperature: Controls the randomness of the model's output. Higher values (e.g., 1.5)
                        make the output more random, while lower values (e.g., 0.2) make it more deterministic.
                        Should be between 0 and 2.
                    tools: List of tool configurations for function calling capabilities
                    stream: Return assistant response as a stream of chunks
                    cached: Caching configuration (Anthropic only)

                Returns:
                    out: Response as text or stream

                Raises:
                    ConnectionError: If unable to reach the server
                    RuntimeError: If rate limit exceeded or API status error
                    Exception: For unexpected errors
                """
                client, messages, role = self._api_helper._set_defaults(
                    model,
                    messages,
                )

                if tools:
                    tools = self._api_helper.normalize_tools(tools)

                self._chat_helper = _ChatHelper(
                    self._api_helper,
                    model,
                    messages,
                    float(temperature),
                    tools,
                    stream,
                    cached,
                    reasoning_effort,
                    client,
                    role
                )

                response = self._chat_helper._get_response()
                if stream:
                    return self._chat_helper._handle_stream(response)
                else:
                    return self._chat_helper._handle_response(response)