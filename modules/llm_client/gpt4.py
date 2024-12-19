from typing import Dict, List, Optional

import aiohttp

from modules.llm_client.interface import LLMClientInterface


class GPT4Client(LLMClientInterface):
    def __init__(self, api_token: str, model: str = "gpt-4-turbo-preview"):
        """
        Initialize the GPT-4 client with the given API token and model.

        Args:
            api_token (str): The API token for authenticating with the OpenAI service.
            model (str): The model to use for generating responses. Defaults to "gpt-4-turbo-preview".
        """
        self.api_token = api_token
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    async def generate(
        self, user_prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response from the GPT-4 model based on the given user and optional system prompts.

        Args:
            user_prompt (str): The prompt provided by the user.
            system_prompt (Optional[str]): An optional system prompt to guide the model's response.

        Returns:
            str: The generated response from the GPT-4 model.

        Raises:
            Exception: If there is a network error, unexpected response format, or other error when calling the OpenAI API.
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.7,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error: {error_text}")

                    response_data = await response.json()
                    return response_data["choices"][0]["message"]["content"]

        except aiohttp.ClientError as e:
            raise Exception(f"Network error when calling OpenAI API: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected response format from OpenAI API: {str(e)}")
        except Exception as e:
            raise Exception(f"Error calling OpenAI API: {str(e)}")
