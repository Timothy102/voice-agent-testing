import requests
import assemblyai as aai

API_ENDPOINT = "https://app.hamming.ai/api/rest/exercise/start-call"
API_TOKEN = "sk-2629df12b920117989d58f6ab10ee710"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

call_id = "cm4u0gs1f00fmirz83y63s9hw"

# Get the recording URL
recording_url = f"https://app.hamming.ai/api/media/exercise?id={call_id}"
recording_response = requests.get(recording_url, headers=headers)

print(recording_response)
if recording_response.status_code == 200:
    print(f"Recording available at: {recording_response}")
    audio_content = recording_response.content
    
    # Save the audio content to a file (optional but recommended for debugging)
    with open('recording.mp3', 'wb') as f:
        f.write(audio_content)

    aai.settings.api_key = "bef0f6e47399485284707cfa65b45bce"
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe("./recording.mp3")
    print(transcript.text)
else:
    print("Recording not yet available")
