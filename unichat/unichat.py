import json
from typing import Dict, Generator, List, Union

import anthropic
import google.generativeai as genai
import openai
from mistralai import Mistral

from .models import MODELS_LIST, MODELS_MAX_TOKEN


class UnifiedChatApi:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._api_helper = _ApiHelper(
            api_key=self.api_key,
        )
        self.chat = self.Chat(self._api_helper)

    class Chat:
        def __init__(self, _api_helper):
            self._api_helper = _api_helper
            self.completions = self.Completions(_api_helper)

        class Completions:
            def __init__(self, _api_helper):
                self._api_helper = _api_helper
                self._chat_helper = None

            def create(
                self,
                model: str,
                messages: List[Dict[str, str]],
                temperature: str = "1.0",
                stream: bool = True,
                cached: Union[bool, str] = False,
            ) -> Union[Generator | str]:
                """
                Get chat completion from various AI models.

                Args:
                    model_name: Name of the model to use
                    messages: List of conversation messages
                    temperature: Controls the randomness of the model's output. Higher values (e.g., 1.5)
                        make the output more random, while lower values (e.g., 0.2) make it more deterministic.
                        Should be between 0 and 2.
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
                    temperature,
                )

                self._chat_helper = _ChatHelper(self._api_helper, model, messages, temperature, stream, cached, client, role)

                response = self._chat_helper._get_response()
                if stream:
                    return self._chat_helper._handle_stream(response)
                else:
                    return self._chat_helper._handle_response(response)

class _ApiHelper:
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, api_key):
        self.api_key = api_key
        self.models = MODELS_LIST
        self.max_tokens = MODELS_MAX_TOKEN
        self.api_client = None

    def _get_max_tokens(self, model_name: str) -> int:
        return int(self.max_tokens.get(model_name, self.DEFAULT_MAX_TOKENS))

    def _get_client(self, model_name: str, temperature: str, role: str = ""):
        if self.api_client is not None:
            return self.api_client

        if model_name in self.models["mistral_models"]:
            client = Mistral(api_key=self.api_key)
        elif model_name in self.models["anthropic_models"]:
            client = anthropic.Anthropic(api_key=self.api_key)
        elif model_name in self.models["grok_models"]:
            client = openai.OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
        elif model_name in self.models["gemini_models"]:
            genai.configure(api_key=self.api_key)
            generation_config = {
                "temperature": float(temperature),
            }
            client = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=role,
                tools="code_execution",
            )
        elif model_name in self.models["openai_models"]:
            client = openai.OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Model '{model_name}' not found.")

        self.api_client = client
        return client

    def _set_defaults(
        self,
        model_name: str,
        conversation: List[Dict[str, str]],
        temperature: str,
    ):
        # Extract the system instructions from the conversation.
        # OpenAI "o1" models do not support system role as part of the beta limitations. More info here: https://platform.openai.com/docs/guides/reasoning/beta-limitations
        if model_name in self.models["anthropic_models"] or model_name in self.models["gemini_models"] or model_name.startswith("o1"):
            role = conversation[0]["content"] if conversation[0]["role"] == "system" else ""
            conversation = [message for message in conversation if message["role"] != "system"]
        else:
            role = ""
        client = self._get_client(model_name, temperature, role)
        return client, conversation, role

class _ChatHelper:
    def __init__(self, api_helper, model_name, messages, temperature, stream, cached, client, role):
        self.api_helper = api_helper
        self.model_name = model_name
        self.messages = messages
        self.temperature = float(temperature)
        self.stream = stream
        self.cached = cached
        self.client = client
        self.role = role

    def _get_response(self) -> any:
        try:
            if self.model_name in self.api_helper.models["mistral_models"]:
                if self.stream:
                    response = self.client.chat.stream(
                        model=self.model_name,
                        temperature=self.temperature,
                        messages=self.messages,
                    )
                else:
                    response = self.client.chat.complete(
                        model=self.model_name,
                        temperature=self.temperature,
                        messages=self.messages,
                    )

            elif self.model_name in self.api_helper.models["anthropic_models"]:
                self.temperature = 1 if self.temperature > 1 else self.temperature
                if self.cached is False:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=self.api_helper._get_max_tokens(self.model_name),
                        temperature=self.temperature,
                        system=self.role,
                        messages=self.messages,
                        stream=self.stream,
                    )
                else:
                    response = self.client.beta.prompt_caching.messages.create(
                        model=self.model_name,
                        max_tokens=self.api_helper._get_max_tokens(self.model_name),
                        temperature=self.temperature,
                        system=[
                            {"type": "text", "text": self.role},
                            {"type": "text", "text": self.cached, "cache_control": {"type": "ephemeral"}},
                        ],
                        messages=self.messages,
                        stream=self.stream,
                    )

            elif self.model_name in self.api_helper.models["gemini_models"]:
                formatted_messages = [
                    {"role": "model" if item["role"] == "assistant" else item["role"], "parts": [item["content"]]}
                    for item in self.messages
                ]
                chat_session = self.client.start_chat(history=formatted_messages[:-1])
                response = chat_session.send_message(formatted_messages[-1]["parts"][0], stream=self.stream)

            elif self.model_name in self.api_helper.models["grok_models"]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    messages=self.messages,
                    stream=self.stream,
                )

            elif self.model_name in self.api_helper.models["openai_models"]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    messages=self.messages,
                    stream=self.stream,
                )

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
            raise Exception(f"An unexpected error occurred: {e}") from e

    def _handle_response(self, response) -> str:
        try:
            if self.model_name in self.api_helper.models["mistral_models"]:
                response_content = response.choices[0].message.content

            elif self.model_name in self.api_helper.models["anthropic_models"]:
                response_content = response.content[0].text

            elif self.model_name in self.api_helper.models["gemini_models"]:
                response_content = response.text

            elif self.model_name in self.api_helper.models["grok_models"]:
                response_content = response.choices[0].message.content

            elif self.model_name in self.api_helper.models["openai_models"]:
                response_content = response.choices[0].message.content

            return response_content

        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}") from e

    def _handle_stream(self, response) -> Generator:
        try:
            if self.model_name in self.api_helper.models["mistral_models"]:
                for chunk in response:
                    yield chunk.data.choices[0].delta.content

            elif self.model_name in self.api_helper.models["anthropic_models"]:
                for chunk in response:
                    if chunk.type == "content_block_delta":
                        yield chunk.delta.text

            elif self.model_name in self.api_helper.models["gemini_models"]:
                for chunk in response:
                    yield chunk.text

            elif self.model_name in self.api_helper.models["grok_models"]:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            elif self.model_name in self.api_helper.models["openai_models"]:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}") from e
