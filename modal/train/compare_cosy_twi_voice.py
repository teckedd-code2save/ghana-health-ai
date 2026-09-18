"""Private Twi-trained CosyVoice listening test with a consented owner reference.

Reference audio is a function argument and temporary file, never a model-volume
artifact. No public endpoint, saved speaker embedding, or voice publication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("ghana-cosy-twi-private-comparison")
MODEL = "neriqlabs/kasanoma-tts-twi-v0.4"
REVISION = "9ef4b2dde5fb4029ec019928725abae7d8eeae6d"
CODE_REVISION = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
cache = modal.Volume.from_name("ghana-health-tts-stable-twi-models", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1", "build-essential")
    .pip_install("setuptools==80.9.0", "wheel==0.45.1", "numpy==1.26.4", "Cython==3.0.12")
    .pip_install("torch==2.3.1", "torchaudio==2.3.1", "numpy==1.26.4", "transformers==4.51.3",
                 "huggingface_hub==0.36.0", "modelscope==1.20.0", "HyperPyYAML==1.2.3",
                 "onnxruntime-gpu==1.18.0", "inflect==7.3.1", "librosa==0.10.2", "soundfile==0.12.1",
                 "openai-whisper==20231117", "conformer==0.3.2", "diffusers==0.29.0",
                 "omegaconf==2.3.0", "hydra-core==1.3.2", "lightning==2.2.4", "pyworld==0.3.4",
                 "pyarrow==18.1.0", "x-transformers==2.11.24", "rich==13.7.1", "gdown==5.1.0",
                 extra_options="--no-build-isolation")
    .run_commands("git clone --no-checkout https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice",
                  f"git -C /opt/CosyVoice checkout {CODE_REVISION}",
                  "git -C /opt/CosyVoice submodule update --init --recursive")
    .env({"PYTHONPATH": "/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS",
          "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"})
    .pip_install("matplotlib==3.7.5", "onnx==1.16.0", "tensorboard==2.14.0", "protobuf==4.25.0")
    .pip_install("wget==3.2")
    .run_commands("python -c 'from cosyvoice.cli.cosyvoice import CosyVoice3; from cosyvoice.flow.flow_matching import CausalConditionalCFM; from cosyvoice.llm.llm import CosyVoice3LM; from cosyvoice.hifigan.generator import HiFTGenerator'")
)
fast_image = (
    image.pip_install("torch==2.5.1", "torchaudio==2.5.1", "onnxruntime-gpu==1.20.2",
                      "openai-whisper==20240930", extra_options="--no-build-isolation")
    .env({"LD_LIBRARY_PATH": ":".join([
        "/usr/local/lib/python3.11/site-packages/nvidia/" + name + "/lib"
        for name in ("cudnn", "cublas", "cuda_runtime", "cuda_nvrtc", "cuda_cupti",
                     "cufft", "curand", "cusolver", "cusparse", "nvjitlink", "nvtx", "nccl")
    ] + ["/usr/local/cuda/lib64", "/usr/local/nvidia/lib64"])})
    .run_commands("python -m pip check",
                  "python -c \"import subprocess; import onnxruntime; from pathlib import Path; "
                  "library = Path(onnxruntime.__file__).parent / 'capi/libonnxruntime_providers_cuda.so'; "
                  "result = subprocess.check_output(['ldd', str(library)], text=True); print(result); "
                  "assert not any('not found' in line and 'libcuda.so' not in line for line in result.splitlines()), "
                  "'Missing CUDA runtime dependency'\"")
)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1200,
              max_containers=1, retries=0, volumes={"/models": cache})
def synthesize(reference: bytes, texts: list[dict], streaming: bool = False, fp16: bool = False):
    import io
    import logging
    import tempfile
    import time
    from datetime import datetime, timezone

    import numpy as np
    import soundfile as sf
    import torch
    import onnxruntime
    from huggingface_hub import snapshot_download
    from cosyvoice.cli.cosyvoice import CosyVoice3

    logging.getLogger().setLevel(logging.WARNING)

    if len(reference) > 1_000_000 or not 1 <= len(texts) <= 3 or any(len(t["text"]) > 300 for t in texts):
        raise ValueError("Comparison budget exceeded")
    samples, rate = sf.read(io.BytesIO(reference))
    if rate != 16000 or samples.ndim != 1 or not 3 <= len(samples)/rate <= 15 or not np.isfinite(samples).all():
        raise ValueError("Expected a clean 3-15 second mono 16kHz reference")
    model_dir = snapshot_download(MODEL, revision=REVISION, cache_dir="/models/cosy-hf",
                                  ignore_patterns=["*.fp32.onnx", "*.batch.onnx", "asset/*"])
    try:
        engine = CosyVoice3(model_dir, load_trt=False, load_vllm=False, fp16=fp16)
    except Exception as exc:
        raise RuntimeError(f"Voice model initialization failed: {exc}") from None
    initial_hop = engine.model.token_hop_len
    providers = engine.frontend.speech_tokenizer_session.get_providers()
    if fp16 and "CUDAExecutionProvider" not in providers:
        raise RuntimeError("GPU speech tokenizer did not initialize; do not claim an optimized result")
    results = []
    with tempfile.TemporaryDirectory() as folder:
        prompt = Path(folder) / "reference.wav"
        prompt.write_bytes(reference)
        for row in texts:
            # The pinned upstream implementation grows this instance field during
            # streaming. Our sequential benchmark must reset it for each utterance.
            if streaming:
                engine.model.token_hop_len = initial_hop
            started = time.monotonic()
            torch.manual_seed(42)
            chunks, timings = [], []
            for chunk in engine.inference_cross_lingual(
                "You are a helpful assistant.<|endofprompt|>" + row["text"], str(prompt),
                stream=streaming, text_frontend=False,
            ):
                chunks.append(chunk)
                timings.append({"seconds": time.monotonic() - started,
                                "audio_seconds": chunk["tts_speech"].shape[-1] / engine.sample_rate})
            if not chunks:
                raise RuntimeError("No speech returned")
            audio = torch.cat([c["tts_speech"].cpu() for c in chunks], dim=1).squeeze().numpy()
            if not np.isfinite(audio).all() or audio.size == 0:
                raise RuntimeError("Invalid generated audio")
            stream = io.BytesIO()
            sf.write(stream, audio, engine.sample_rate, format="WAV", subtype="PCM_16")
            results.append({**row, "voice": "cosy-twi-owner-reference", "model": MODEL, "revision": REVISION,
                            "sample_rate": engine.sample_rate, "duration": len(audio)/engine.sample_rate,
                            "seconds": time.monotonic()-started, "wav": stream.getvalue(),
                            "streaming": streaming, "chunk_timings": timings,
                            "initial_token_hop": initial_hop, "final_token_hop": engine.model.token_hop_len,
                            "streaming_state_reset": streaming,
                            "first_audio_seconds": timings[0]["seconds"]})
    cache.commit()
    return {"created": datetime.now(timezone.utc).isoformat(), "code_revision": CODE_REVISION,
            "runtime": {"torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
                        "fp16": fp16, "speech_tokenizer_providers": providers},
            "reference_sha256": hashlib.sha256(reference).hexdigest(), "results": results}


@app.function(image=fast_image, gpu="L4", cpu=4, memory=16384, timeout=1200,
              max_containers=1, retries=0, volumes={"/models": cache})
def synthesize_fast(reference: bytes, texts: list[dict]):
    return synthesize.local(reference, texts, streaming=True, fp16=True)


@app.function(image=fast_image, gpu="L4", cpu=4, memory=16384, timeout=1200,
              max_containers=1, retries=0, volumes={"/models": cache})
def compare_tail(reference: bytes, text: dict):
    import gc
    import torch

    reports = []
    for label, streaming, fp16 in (("control-fp16-stream", True, True),
                                   ("fp32-stream", True, False),
                                   ("fp16-whole", False, True)):
        report = synthesize.local(reference, [text], streaming=streaming, fp16=fp16)
        report["variant"] = label
        reports.append(report)
        gc.collect()
        torch.cuda.empty_cache()
    return reports


@app.local_entrypoint()
def main(streaming: bool = False, optimized: bool = False, tail_check: bool = False):
    root = Path(__file__).resolve().parents[2]
    folder = root / "tmp/twi-voice-comparison"
    reference_path = folder / "speaker-1-reference.wav"
    manifest = json.loads((folder / "manifest.json").read_text())
    texts = [{"id": r["id"], "text": r["text"]} for r in manifest["results"] if r["voice"] == "nano-4step"]
    if tail_check:
        if streaming or optimized:
            raise ValueError("Tail comparison has its own fixed runtime variants")
        output = folder / "cosy-tail-check.json"
        if output.exists() or list(folder.glob("tail-*.wav")):
            raise ValueError("Tail comparison already exists; do not overwrite listening evidence")
        text = next(row for row in texts if row["id"] == "question")
        reports = compare_tail.remote(reference_path.read_bytes(), text)
        for report in reports:
            for row in report["results"]:
                wav = row.pop("wav")
                row["file"] = f"tail-{report['variant']}-{row['id']}.wav"
                row["sha256"] = hashlib.sha256(wav).hexdigest()
                with (folder / row["file"]).open("xb") as audio:
                    audio.write(wav)
        result = {"purpose": "Investigate owner-reported muffled ending; same reference, text, seed and optimized runtime",
                  "limitations": ["Different execution modes can change generated tokens; not a token-controlled vocoder experiment.",
                                  "No automatic perceptual quality claim; requires listening.",
                                  "Original comparison samples remain unchanged."], "reports": reports}
        with output.open("x") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    report = (synthesize_fast.remote(reference_path.read_bytes(), texts) if optimized
              else synthesize.remote(reference_path.read_bytes(), texts, streaming=streaming))
    prefix = "stream-fp16-" if optimized else "stream-reset-" if streaming else ""
    for row in report["results"]:
        wav = row.pop("wav")
        row["file"] = prefix + f"{row['id']}-{row['voice']}.wav"
        row["sha256"] = hashlib.sha256(wav).hexdigest()
        (folder / row["file"]).write_bytes(wav)
    report["reference_source"] = "Owner-provided speaker-1 recording health_twi_sp001_u0005.webm; prior explicit voice-use authorization."
    report["quality_status"] = "Unrated private samples; not a trained or published owner voice."
    filename = "cosy-streaming-fp16.json" if optimized else "cosy-streaming-reset.json" if streaming else "cosy-manifest.json"
    (folder / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
