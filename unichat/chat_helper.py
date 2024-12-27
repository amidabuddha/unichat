from typing import Generator, List, Any, Optional, Union, Dict, Any
import anthropic
import openai


class _ChatHelper:
    def __init__(
        self,
        api_helper,
        model_name: str,
        messages: List[dict],
        temperature: float,
        tools: Optional[List[dict]] = None,
        stream: bool = False,
        cached: Union[bool, str] = False,
        client: Any = None,
        role: str = ""
    ):
        self.api_helper = api_helper
        self.model_name = model_name
        self.messages = messages
        self.temperature = temperature
        self.tools = tools or []
        self.stream = stream
        self.cached = cached
        self.client = client
        self.role = role

    def _get_response(self) -> Any:
        try:
            if self.model_name in self.api_helper.models["mistral_models"]:
                mistral_params = {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "messages": self.messages,
                }
                if self.tools:
                    mistral_params["tools"] = self.api_helper.transform_tools(self.tools)

                if self.stream:
                    response = self.client.chat.stream(**mistral_params)
                else:
                    response = self.client.chat.complete(**mistral_params)

            elif self.model_name in self.api_helper.models["anthropic_models"]:
                self.temperature = 1 if self.temperature > 1 else self.temperature
                anthropic_messages = self.api_helper.transform_messages(self.messages)
                anthropic_params = {
                    "model": self.model_name,
                    "max_tokens": self.api_helper._get_max_tokens(self.model_name),
                    "temperature": self.temperature,
                    "stream": self.stream,
                }

                if self.tools:
                    self.tools[-1].update({"cache_control": {"type": "ephemeral"}})
                    anthropic_params["tools"] = self.tools

                if self.cached is False:
                    anthropic_params["system"] = self.role
                else:
                    anthropic_params["system"] = [
                        {"type": "text", "text": self.role},
                        {"type": "text", "text": self.cached, "cache_control": {"type": "ephemeral"}},
                    ]

                anthropic_params["messages"] = self.api_helper.cache_messages(anthropic_messages)

                response = self.client.messages.create(**anthropic_params)

            elif self.model_name in (
                self.api_helper.models["gemini_models"]
                + self.api_helper.models["grok_models"]
                + self.api_helper.models["deepseek_models"]
                + self.api_helper.models["openai_models"]
            ):
                params = {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "messages": self.messages,
                    "stream": self.stream,
                }

                if self.tools and not self.model_name.startswith("o1"):
                    params["tools"] = self.api_helper.transform_tools(self.tools)

                response = self.client.chat.completions.create(**params)

            else:
                raise ValueError(f"Model {self.model_name} is currently not supported")

            return response

        except (openai.APIConnectionError, anthropic.APIConnectionError) as e:
            raise ConnectionError(f"The server could not be reached: {e}") from e
        except (openai.RateLimitError, anthropic.RateLimitError) as e:
            raise RuntimeError(f"Rate limit exceeded: {e}") from e
        except (openai.APIStatusError, anthropic.APIStatusError, anthropic.BadRequestError) as e:
            raise RuntimeError(f"API status error: {e.status_code} - {e.message}") from e
        except Exception as e:
            raise Exception(f"An unexpected error occurred at _get_response {e}") from e

    def _handle_response(self, response) -> Any:
        """Handle non-streaming response."""
        try:
            if self.model_name in self.api_helper.models["anthropic_models"]:
                return self.api_helper.convert_claude_to_gpt(response)
            elif self.model_name in self.api_helper.models["mistral_models"]:
                return self.api_helper.transform_response(response)
            else:
                return response
        except Exception as e:
            raise Exception(f"An unexpected error occurred at _handle_response: {e}") from e

    def _handle_stream(self, response) -> Generator[Dict[str, Any], None, None]:
        """Handle streaming response."""
        try:
            if self.model_name in self.api_helper.models["anthropic_models"]:
                for chunk in self.api_helper.transform_stream(response):
                    if chunk:
                        yield chunk

            elif self.model_name in self.api_helper.models["mistral_models"]:
                for chunk in self.api_helper.transform_stream_chunk(response):
                    if chunk:
                        yield chunk

            else:
                for chunk in response:
                    if chunk:
                        yield chunk

        except Exception as e:
            raise Exception(f"An unexpected error occurred while streaming: {e}") from e