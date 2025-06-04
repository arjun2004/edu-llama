api_key = "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"
assistant = AILearningAssistant(api_key)
client = assistant.client
if client.validate_api_key():
    print("API key works!")
else:
    print("API key has issues")