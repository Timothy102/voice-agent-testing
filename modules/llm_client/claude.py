import os
from typing import Optional

import aiohttp

from modules.llm_client.interface import LLMClientInterface


class ClaudeClient(LLMClientInterface):
    def __init__(self, api_token: str, model: str = "claude-3-5-sonnet-20240620"):
        """
        Initialize the Claude client with the given API token and model.

        Args:
            api_token (str): The API token for authenticating with the Anthropic service.
            model (str): The model to use for generating responses. Defaults to "claude-3-sonnet-20240229".
        """
        self.api_token = api_token
        self.model = model

    async def generate(
        self, user_prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response from the Claude model based on the given user and optional system prompts.

        Args:
            user_prompt (str): The prompt provided by the user.
            system_prompt (Optional[str]): An optional system prompt to guide the model's response.

        Returns:
            str: The generated response from the Claude model.

        Raises:
            Exception: If there is a network error, unexpected response format, or other error when calling the Anthropic API.
        """
        headers = {
            "x-api-key": self.api_token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        messages = [{"role": "user", "content": user_prompt}]

        payload = {
            "model": self.model,
            "max_tokens": 1500,
            "temperature": 0,
            "messages": messages,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Anthropic API error: {error_text}")

                    response_data = await response.json()
                    return response_data["content"][0]["text"]

        except aiohttp.ClientError as e:
            raise Exception(f"Network error when calling Anthropic API: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected response format from Anthropic API: {str(e)}")
        except Exception as e:
            raise Exception(f"Error calling Anthropic API: {str(e)}")
