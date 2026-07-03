import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
with open("/root/repos/hermes/docs/code_asr_awq/quant/test_asr_long.wav", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

ok = 0
fail = 0
for attempt in range(20):
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-ASR-1.7B", stream=True,
        messages=[{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + b64}}]}],
        stream_options={"include_usage": True},
    )
    parts = []
    ot = 0
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
        if hasattr(chunk, "usage") and chunk.usage:
            ot = chunk.usage.completion_tokens
    txt = "".join(parts)
    if len(txt) > 10:
        ok += 1
    else:
        fail += 1
    print(f"Run {attempt+1:2d}: {'OK' if len(txt)>10 else 'FAIL'} chars={len(txt):4d} tokens={ot:3d}")

print(f"\nOK={ok} FAIL={fail} rate={fail/(ok+fail)*100:.0f}%")
