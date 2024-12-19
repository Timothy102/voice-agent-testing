from abc import ABC, abstractmethod


class PhoneInterface(ABC):
    """Abstract base class defining the interface for phone call functionality."""

    @abstractmethod
    async def make_call(self, phone_number: str, prompt: str) -> int:
        """Make an outbound phone call.

        Args:
            phone_number (str): The phone number to call
            prompt (str): The initial prompt/message to use

        Returns:
            int: Unique identifier for the call
        """
        pass

    @abstractmethod
    async def poll_for_response(
        self, call_id: int, time_in_between_polls: int = 200, max_time: int = 480
    ) -> str:
        """Poll the API for a response recording.

        Args:
            call_id (int): ID of the call to poll for
            time_in_between_polls (int): Time to wait between polls in seconds. Defaults to 200.
            max_time (int): Maximum time to poll in seconds before raising a TimeoutError. Defaults to 480.

        Returns:
            str: Path to the downloaded audio file

        Raises:
            TimeoutError: If max_time is exceeded
            ValueError: If server returns an error
        """
        pass
