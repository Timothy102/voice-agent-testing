import asyncio
from pathlib import Path

import aiofiles
import aiohttp

from modules.transcriber.interface import TranscriberInterface


class AssemblyAITranscriber(TranscriberInterface):
    """Transcriber implementation using the AssemblyAI API for speech-to-text conversion.

    This class handles uploading audio files, requesting transcriptions, and polling for results
    from the AssemblyAI service.

    Args:
        api_token (str): Authentication token for the AssemblyAI API
    """

    def __init__(self, api_token: str):
        """Initialize the transcriber with API credentials.

        Args:
            api_token (str): Authentication token for the AssemblyAI API
        """
        super().__init__(api_token)
        self.base_url = "https://api.assemblyai.com/v2"

    async def transcribe(self, filepath: Path) -> str:
        """Transcribe an audio file to text using AssemblyAI.

        This method handles the full transcription workflow:
        1. Upload the audio file
        2. Request transcription
        3. Poll for results

        Args:
            filepath (Path): Path to the audio file to transcribe

        Returns:
            str: The transcribed text

        Raises:
            Exception: If any step of the transcription process fails
        """
        headers = {"authorization": self.api_token, "content-type": "application/json"}

        async with aiofiles.open(filepath, "rb") as file:
            file_data = await file.read()

        async with aiohttp.ClientSession() as session:
            # Upload the audio file
            async with session.post(
                f"{self.base_url}/upload", headers=headers, data=file_data
            ) as upload_response:
                if upload_response.status != 200:
                    error_text = await upload_response.text()
                    raise Exception(f"AssemblyAI upload error: {error_text}")

                upload_data = await upload_response.json()
                audio_url = upload_data["upload_url"]

            # Request transcription
            payload = {"audio_url": audio_url}

            async with session.post(
                f"{self.base_url}/transcript", headers=headers, json=payload
            ) as transcript_response:
                if transcript_response.status != 200:
                    error_text = await transcript_response.text()
                    raise Exception(f"AssemblyAI transcription error: {error_text}")

                transcript_data = await transcript_response.json()
                transcript_id = transcript_data["id"]

            # Poll for transcription result
            while True:
                async with session.get(
                    f"{self.base_url}/transcript/{transcript_id}", headers=headers
                ) as result_response:
                    if result_response.status != 200:
                        error_text = await result_response.text()
                        raise Exception(
                            f"AssemblyAI result polling error: {error_text}"
                        )

                    result_data = await result_response.json()
                    if result_data["status"] == "completed":
                        return result_data["text"]
                    elif result_data["status"] == "failed":
                        raise Exception(
                            f"AssemblyAI transcription failed: {result_data['error']}"
                        )

                await asyncio.sleep(10)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    async def test_transcription():
        """Test function to demonstrate transcription functionality.

        Loads API credentials from environment and transcribes a test audio file.
        """
        # Load environment variables
        load_dotenv()
        api_token = os.getenv("ASSEMBLYAI_API_TOKEN")

        # Initialize transcriber
        transcriber = AssemblyAITranscriber(api_token=api_token)

        # Transcribe test file
        audio_file = "transcripts/cm4u5chwj00luirz8bepiwfzc.mp3"
        transcript = await transcriber.transcribe(audio_file)
        print(f"Transcript: {transcript}")

    # Run test
    asyncio.run(test_transcription())
