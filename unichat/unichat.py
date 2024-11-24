import json
from typing import Dict, List, Union

import anthropic
import google.generativeai as genai
import openai
from mistralai import Mistral

from .models import MODELS_LIST, MODELS_MAX_TOKEN


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

class UnifiedChatApi:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.helper = _ApiHelper(
            api_key=self.api_key,
        )
        self.chat = self.Chat(self.helper)

    class Chat:
        def __init__(self, helper):
            self.helper = helper
            self.completions = self.Completions(helper)

        class Completions:
            def __init__(self, helper):
                self.helper = helper

            def create(
                self,
                model_name: str,
                messages: List[Dict[str, str]],
                temperature: str = "1.0",
                cached: Union[bool, str] = False,
            ) -> str:
                """
                Get chat completion from various AI models.

                Args:
                    model_name: Name of the model to use
                    messages: List of conversation messages
                    temperature: Controls the randomness of the model's output. Higher values (e.g., 1.5)
                        make the output more random, while lower values (e.g., 0.2) make it more deterministic.
                        Should be between 0 and 2.
                    cached: Caching configuration (Anthropic only)

                Returns:
                    str: The generated response

                Raises:
                    ConnectionError: If unable to reach the server
                    RuntimeError: If rate limit exceeded or API status error
                    Exception: For unexpected errors
                """
                client, messages, role = self.helper._set_defaults(
                    model_name,
                    messages,
                    temperature,
                )
                try:
                    if model_name in self.helper.models["mistral_models"]:
                        response = client.chat.complete(
                            model=model_name,
                            temperature=float(temperature),
                            messages=messages,
                        )
                        response_content = response.choices[0].message.content

                    elif model_name in self.helper.models["anthropic_models"]:
                        temperature = 1 if float(temperature) > 1 else temperature
                        if cached is False:
                            response = client.messages.create(
                                model=model_name,
                                max_tokens=self.helper._get_max_tokens(model_name),
                                temperature=float(temperature),
                                system=role,
                                messages=messages,
                            ).model_dump_json()
                        else:
                            response = client.beta.prompt_caching.messages.create(
                                model=model_name,
                                max_tokens=self.helper._get_max_tokens(model_name),
                                temperature=float(temperature),
                                system=[
                                    {"type": "text", "text": role},
                                    {"type": "text", "text": cached, "cache_control": {"type": "ephemeral"}},
                                ],
                                messages=messages,
                            ).model_dump_json()

                        response = json.loads(response)
                        response_content = response["content"][0]["text"]

                    elif model_name in self.helper.models["gemini_models"]:
                        formatted_messages = [
                            {"role": "model" if item["role"] == "assistant" else item["role"], "parts": [item["content"]]}
                            for item in messages
                        ]
                        chat_session = client.start_chat(history=formatted_messages[:-1])
                        response = chat_session.send_message(formatted_messages[-1]["parts"][0])
                        response_content = response.text

                    elif model_name in self.helper.models["grok_models"]:
                        response = client.chat.completions.create(
                            model=model_name,
                            temperature=float(temperature),
                            messages=messages,
                        )
                        response_content = response.choices[0].message.content

                    elif model_name in self.helper.models["openai_models"]:
                        response = client.chat.completions.create(
                            model=model_name,
                            temperature=float(temperature),
                            messages=messages,
                        )
                        response_content = response.choices[0].message.content

                    else:
                        return f"Model {model_name} is currently not supported"

                    return response_content

                except (openai.APIConnectionError, anthropic.APIConnectionError) as e:
                    raise ConnectionError(f"The server could not be reached: {e}") from e
                except (openai.RateLimitError, anthropic.RateLimitError) as e:
                    raise RuntimeError(f"Rate limit exceeded: {e}") from e
                except (openai.APIStatusError, anthropic.APIStatusError, anthropic.BadRequestError) as e:
                    raise RuntimeError(f"API status error: {e.status_code} - {e.message}") from e
                except Exception as e:
                    raise Exception(f"An unexpected error occurred: {e}") from e