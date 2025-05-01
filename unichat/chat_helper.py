import json
from typing import Any, Dict, Generator, List, Optional, Union

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
        reasoning_effort: Union[bool, str] = False,
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
        self.reasoning_effort = reasoning_effort
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
                anthropic_messages = self.api_helper.transform_messages(self.messages)
                self.api_helper.anthropic_conversation.append(anthropic_messages[-1])
                anthropic_params = {
                    "model": self.model_name,
                    "max_tokens": self.api_helper._get_max_tokens(self.model_name),
                    "temperature": 1 if self.reasoning_effort else min(self.temperature, 1),
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

                match self.reasoning_effort:
                    case "high":
                        anthropic_params["thinking"] = {"type": "enabled", "budget_tokens": int(anthropic_params["max_tokens"]*0.8)}
                    case "medium":
                        anthropic_params["thinking"] = {"type": "enabled", "budget_tokens": int(anthropic_params["max_tokens"]*0.5)}
                    case "low":
                        anthropic_params["thinking"] = {"type": "enabled", "budget_tokens": int(anthropic_params["max_tokens"]*0.2)}
                    case "none":
                        pass
                    case False:
                        pass
                    case _:
                        raise ValueError(f"Invalid reasoning_effort value: {self.reasoning_effort}")

                anthropic_params["messages"] = self.api_helper.cache_messages(self.api_helper.anthropic_conversation)

                response = self.client.messages.create(**anthropic_params)

            else:
                params = {
                    "model": self.model_name,
                    "messages": self.messages,
                    "stream": self.stream,
                }
                if self.model_name not in ("o1", "o3-mini", "o3", "o4-mini") and not self.model_name.endswith("reasoner") and not self.model_name.startswith("mercury"):
                    params["temperature"] = self.temperature
                if self.tools and not self.model_name.endswith("reasoner"):
                    params["tools"] = self.api_helper.transform_tools(self.tools)
                if self.reasoning_effort:
                    params["reasoning_effort"] = self.reasoning_effort

                response = self.client.chat.completions.create(**params)

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
                content_list = [self.api_helper.block_to_dict(item) for item in response.content]
                self.api_helper.anthropic_conversation.append({
                    "role": "assistant",
                    "content": content_list
                })
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
                # Initialize the message object to collect original blocks
                message = {"role": "assistant", "content": []}

                for transformed_chunk, original_block in self.api_helper.transform_stream(response):
                    if transformed_chunk:
                        yield transformed_chunk  # Yield transformed chunks as needed
                    if original_block:
                        message = self.api_helper.append_block_to_message(message, original_block)

                # After processing, append the message to the conversation
                for block in message["content"]:
                    if block["type"] == "tool_use" and isinstance(block["input"], str):
                        try:
                            block["input"] = json.loads(block["input"])
                        except json.JSONDecodeError:
                            block["input"] = {}
                self.api_helper.anthropic_conversation.append(message)

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