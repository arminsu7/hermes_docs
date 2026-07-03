import base64
import httpx
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

local_audio_path = "./test_asr_long.wav"
with open(local_audio_path, "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

# Create multimodal chat completion request (streaming — vLLM 0.23 非流式 ASR 有 bug)
stream = client.chat.completions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{audio_base64}"
                    }
                }
            ]
        }
    ],
    stream=True,
)

# 收集流式 delta
text_parts = []
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
        text_parts.append(chunk.choices[0].delta.content)

print("".join(text_parts))
