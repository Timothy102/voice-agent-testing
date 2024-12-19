import requests

# API configuration
API_ENDPOINT = "https://app.hamming.ai/api/rest/exercise/start-call"
API_TOKEN = "sk-2629df12b920117989d58f6ab10ee710"

# Call details
phone_number = "+1 (650) 879-8564"  # Auto dealership
prompt = """I'm interested in buying a used BMW. I'd like to know what models you have available 
and what the price ranges are. I'm particularly interested in models from the last 5 years."""

# Make API call to start phone call
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "phone_number": phone_number,
    "prompt": prompt,
    "webhook_url": "https://your-webhook-url.com/callback"  # Replace with actual webhook URL
}

response = requests.post(API_ENDPOINT, json=payload, headers=headers)
call_id = response.json()["id"]

print(f"Call initiated with ID: {call_id}")