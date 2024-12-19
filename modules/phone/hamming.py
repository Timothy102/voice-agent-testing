import asyncio
from abc import ABC, abstractmethod
from typing import Optional

import aiofiles
import aiohttp
import requests

from modules.logger.factory import LoggerFactory
from modules.phone.interface import PhoneInterface


class HammingClient(PhoneInterface):
    """Implementation of PhoneInterface using the Hamming AI API."""

    def __init__(self, api_token: str, api_endpoint: str) -> None:
        """Initialize the Hamming phone client.

        Args:
            settings: Application settings containing API credentials
            logger: Logger instance for tracking execution
        """
        self.api_token = api_token
        self.api_endpoint = api_endpoint
        self.logger = LoggerFactory()
        self.phone_number: Optional[str] = None
        self.prompt: Optional[str] = None

    async def make_call(self, phone_number: str, prompt: str) -> int:
        """Make an outbound phone call via Hamming AI.

        Args:
            phone_number: The phone number to call
            prompt: The prompt/message to use

        Returns:
            int: Unique identifier for the call
        """
        """Makes a call to the Hamming API Endpoint.

        Returns:
            The call ID from the Hamming API response
        """
        self.phone_number = phone_number
        self.prompt = prompt

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "phone_number": self.phone_number,
            "prompt": self.prompt,
            "webhook_url": "https://your-webhook-url.com/callback",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_endpoint, json=payload, headers=headers
            ) as response:
                self.logger.info(
                    f"INFO: Calling phone number {self.phone_number} via API call to {self.api_token}"
                )
                response_data = await response.json()
                return response_data["id"]

    async def poll_for_response(
        self, call_id: int, time_in_between_polls: int = 200, max_time: int = 480
    ) -> str:
        """Poll the API for a response recording.

        Args:
            call_id: ID of the call to poll for from Hamming AI
            time_in_between_polls: Time to wait between polls in seconds
            max_time: Maximum time to poll in seconds before raising a TimeoutError

        Returns:
            Path to the downloaded audio file

        Raises:
            TimeoutError: If max_time is exceeded
            ValueError: If server returns an error
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        recording_url = f"https://app.hamming.ai/api/media/exercise?id={call_id}"
        self.logger.info(f"INFO: Starting to poll for response with call ID: {call_id}")

        start_time = asyncio.get_event_loop().time()
        while True:
            recording_response = requests.get(recording_url, headers=headers)

            if recording_response.status_code == 200:
                audio_content = recording_response.content

                filepath = f"transcripts/{call_id}.mp3"
                async with aiofiles.open(filepath, "wb") as f:
                    await f.write(audio_content)

                self.logger.info(f"INFO: Successfully downloaded audio to {filepath}")
                return filepath

            elif recording_response.status_code >= 500:
                raise ValueError(
                    f"Server error {recording_response.status_code}, retrying..."
                )

            if asyncio.get_event_loop().time() - start_time > max_time:
                raise TimeoutError(
                    f"Recording not available after {max_time} minutes of polling"
                )

            await asyncio.sleep(time_in_between_polls)
