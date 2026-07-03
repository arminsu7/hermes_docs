"""
ASR performance benchmark (OpenAI HTTP API)

Measures: prefill speed, decoder speed, latency, RTF, audio duration,
chars per second, audio throughput.

Usage:
    python test-metrics.py --wav ./test_asr_short.wav --runs 1
    python test-metrics.py --wav ./test_asr_long.wav --runs 5 --warmup-runs 1
"""

import argparse, base64, math, os, time
from dataclasses import dataclass
import soundfile as sf
from openai import OpenAI

DEFAULT_WAV = "./test_asr_long.wav"
DEFAULT_RUNS = 1
DEFAULT_WARMUP_RUNS = 0
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"


@dataclass
class ASRMetrics:
    audio_duration: float = 0.0
    audio_file: str = ""
    total_latency: float = 0.0
    ttft: float = 0.0
    decode_time: float = 0.0
    output_text: str = ""
    output_chars: int = 0
    output_tokens: int = 0
    prompt_tokens: int = 0
    chars_per_sec: float = 0.0
    audio_throughput: float = 0.0
    rtf: float = 0.0
    prefill_speed: float = 0.0
    decoder_speed: float = 0.0

    def __str__(self):
        return (
            f"Audio file:       {self.audio_file}\n"
            f"Audio duration:   {self.audio_duration:.2f}s\n"
            f"Total latency:    {self.total_latency:.2f}s\n"
            f"TTFT (prefill):   {self.ttft:.2f}s\n"
            f"Decode time:      {self.decode_time:.2f}s\n"
            f"Output chars:     {self.output_chars}\n"
            f"Output tokens:    {self.output_tokens}\n"
            f"Prompt tokens:    {self.prompt_tokens}\n"
            f"Prefill speed:    {self.prefill_speed:.1f} tok/s\n"
            f"Decoder speed:    {self.decoder_speed:.1f} tok/s\n"
            f"Chars per sec:    {self.chars_per_sec:.1f}\n"
            f"Audio throughput: {self.audio_throughput:.2f}x realtime\n"
            f"RTF:              {self.rtf:.4f}"
        )


def get_audio_duration(wav_path):
    return sf.info(wav_path).duration


def encode_audio_base64(wav_path):
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def percentile(values, p):
    if not values:
        return 0.0
    sv = sorted(values)
    if len(sv) == 1:
        return sv[0]
    rank = (len(sv) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sv) - 1)
    w = rank - lo
    return sv[lo] + (sv[hi] - sv[lo]) * w


def run_single_asr(client, wav_path, verbose=False):
    audio_duration = get_audio_duration(wav_path)
    audio_base64 = encode_audio_base64(wav_path)

    m = ASRMetrics(
        audio_duration=audio_duration,
        audio_file=os.path.basename(wav_path),
    )

    t0 = time.perf_counter()
    first_at = None

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [{
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}
            }]
        }],
        stream=True,
        stream_options={"include_usage": True},
    )

    text_parts = []
    for chunk in stream:
        now = time.perf_counter()
        if first_at is None:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                first_at = now
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            text_parts.append(chunk.choices[0].delta.content)
        if hasattr(chunk, "usage") and chunk.usage:
            m.prompt_tokens = chunk.usage.prompt_tokens
            m.output_tokens = chunk.usage.completion_tokens

    t1 = time.perf_counter()
    m.output_text = "".join(text_parts)
    m.output_chars = len(m.output_text)
    m.total_latency = t1 - t0
    m.ttft = (first_at - t0) if first_at else m.total_latency
    m.decode_time = m.total_latency - m.ttft
    m.chars_per_sec = m.output_chars / m.total_latency if m.total_latency > 0 else 0.0
    m.audio_throughput = audio_duration / m.total_latency if m.total_latency > 0 else 0.0
    m.rtf = m.total_latency / audio_duration if audio_duration > 0 else 0.0
    m.prefill_speed = m.prompt_tokens / m.ttft if m.ttft > 0 else 0.0
    m.decoder_speed = m.output_tokens / m.decode_time if m.decode_time > 0 else 0.0

    if verbose:
        print(m)
        print("-" * 60)

    return m


def print_summary(runs, warmup_runs):
    lat = [r.total_latency for r in runs]
    rtfs = [r.rtf for r in runs]
    tputs = [r.audio_throughput for r in runs]
    css = [r.chars_per_sec for r in runs]
    ps = [r.prefill_speed for r in runs]
    ds = [r.decoder_speed for r in runs]
    ttfts = [r.ttft for r in runs]
    dts = [r.decode_time for r in runs]
    ocs = [r.output_chars for r in runs]
    ots = [r.output_tokens for r in runs]

    def stat(name, vals, unit="", fmt=".2f"):
        avg = sum(vals) / len(vals)
        p50 = percentile(vals, 50)
        p95 = percentile(vals, 95)
        print(f"{name:<20} avg={avg:{fmt}}{unit}  p50={p50:{fmt}}{unit}  "
              f"p95={p95:{fmt}}{unit}  min={min(vals):{fmt}}{unit}  max={max(vals):{fmt}}{unit}")

    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Audio file:      {runs[0].audio_file}")
    print(f"Audio duration:  {runs[0].audio_duration:.2f}s")
    print(f"Warmup runs:     {warmup_runs}")
    print(f"Benchmark runs:  {len(runs)}")
    print()
    print("--- Timing ---")
    stat("Total latency", lat, "s", ".3f")
    stat("TTFT (prefill)", ttfts, "s", ".3f")
    stat("Decode time", dts, "s", ".3f")
    print()
    print("--- Speed ---")
    stat("Prefill speed", ps, " tok/s", ".1f")
    stat("Decoder speed", ds, " tok/s", ".1f")
    stat("Chars per sec", css, "", ".1f")
    print()
    print("--- Efficiency ---")
    stat("Audio throughput", tputs, "x realtime", ".2f")
    stat("RTF", rtfs, "", ".4f")
    print()
    print("--- Output ---")
    avg_oc = sum(ocs) / len(ocs)
    avg_ot = sum(ots) / len(ots)
    print(f"Output chars:    avg={avg_oc:.1f}  min={min(ocs)}  max={max(ocs)}")
    print(f"Output tokens:   avg={avg_ot:.1f}  min={min(ots)}  max={max(ots)}")
    print()
    print("--- Sample (run 1) ---")
    print(runs[0].output_text[:200] + ("..." if len(runs[0].output_text) > 200 else ""))


def main():
    parser = argparse.ArgumentParser(description="ASR performance benchmark")
    parser.add_argument("--wav", default=DEFAULT_WAV)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    for i in range(args.warmup_runs):
        print(f"Warmup {i+1}/{args.warmup_runs}...", end=" ", flush=True)
        run_single_asr(client, args.wav, verbose=False)
        print("done")

    runs = []
    for i in range(args.runs):
        print(f"Run {i+1}/{args.runs}...")
        runs.append(run_single_asr(client, args.wav, verbose=not args.quiet))

    print_summary(runs, args.warmup_runs)


if __name__ == "__main__":
    main()
