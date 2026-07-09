# Ghana Health AI: Deep Technical Specification & Handover Document

**Project Title**: Ghana Health AI – Voice-First Health Companion in Local Ghanaian Dialects with Real-Time Speaker ID & Ecommerce Capabilities  
**Target Users**: Ghanaians in rural/urban areas, especially women (maternal health), elderly, low-literacy users, and small business owners.  
**Core Languages**: Twi (Akan/Asante/Fante), Ga, Ewe, Dagbani (priority), with extensibility to others.  
**Key Features**:
- Real-time voice chat in local dialects (streaming ASR + diarization + speaker verification/"Voice ID").
- Culturally-grounded health advice, symptom checking, maternal health support, mental health triage (with disclaimers).
- Ecommerce integration: Voice shopping for essentials (food, medicine, household), market price checks, order placement in local languages.
- Privacy-first: On-device options + secure cloud (Modal), end-to-end encryption, compliance with Ghana Data Protection Act.
- Offline-capable core (edge models) + seamless cloud fallback.

**Handover Note for Grok Build / Development Team**: This document is a complete, self-contained technical blueprint. It includes architecture, data sources, model choices, training pipelines (leveraging user's $200+ Modal credits), code skeletons, evaluation plan, and phased roadmap. Use it directly to implement or hand to engineers. All datasets and Modal examples are production-ready or near-ready. Focus on Twi first for MVP (highest impact + data availability).

**Version**: 1.0 | **Date**: July 2026 | **Author**: Grok (xAI) in collaboration with user (Accra-based)  
**Status**: Ready for Build – Prioritize MVP in Twi with maternal health + basic voice chat.

---

## 1. Executive Summary & Impact

Ghana has rich linguistic diversity (Twi spoken by ~8-10M+, major dialects) but limited voice AI in local languages, especially for critical domains like health. Current solutions (English-only chatbots or generic Whisper) fail on code-mixed speech, accents, and cultural context, leading to low adoption in rural areas and among women/elderly.

This system solves that with:
- **Fine-tuned ASR** on Ghana-specific datasets (UGSpeechData ~1000h/language, GhanaNLP corpora, AfriSpeech-200 medical accents).
- **Real-time multi-speaker pipeline** (NVIDIA Parakeet + Sortformer on Modal – sub-second latency, speaker labels without enrollment for group conversations).
- **Voice ID / Speaker Verification** for personalized health records (enroll once, verify voice for privacy).
- **Health LLM/RAG** fine-tuned or RAG-augmented on GAIN maternal health Q&A datasets (20k+ code-mixed Twi-English pairs) + local medical guidelines.
- **Ecommerce layer**: Intent classifier + voice commerce (search products, check prices in Twi, voice checkout) – huge for informal traders and rural users.
- **Modal-powered infrastructure**: Serverless GPU training/inference (use your credits for fine-tunes + serving). Edge deployment for offline/low-data use.

**Expected Impact**:
- Improve health access (maternal mortality reduction via timely advice in local tongue).
- Economic empowerment via voice ecommerce.
- Preserve/promote Ghanaian languages in AI.
- Model for other African countries.

**MVP Scope (3-4 months)**: Twi-only, maternal health focus + basic voice chat + simple ecommerce (market prices + voice search). Deploy as mobile PWA + web demo.

**Success Metrics**: WER <25% on Ghana test sets (target <15% post-fine-tune), end-to-end latency <800ms, user retention >40% in pilot, clinical validation score.

---

## 2. High-Level Architecture

### 2.1 Overall Pipeline (Real-time Voice → Action)

```
User Microphone (Mobile/Web)
       ↓ (WebSocket / gRPC streaming, 16kHz PCM or Opus)
[Edge / Client] Optional: Local VAD + lightweight ASR (Whisper.cpp or ONNX Distil-Whisper for offline)
       ↓
[Modal Cloud - GPU]
  1. Streaming ASR (Parakeet Multi-talker or optimized Whisper)
  2. Real-time Speaker Diarization (Sortformer) → "Speaker 1 (User): ... Speaker 2 (Doctor/Family): ..."
  3. Speaker Verification / Voice ID (embeddings + cosine sim for enrolled users)
  4. Language/Dialect ID + Normalization (code-mixed Twi-English handling)
  5. Intent Classification (Health vs Ecommerce vs General)
       ↓
[Health Branch]
  - RAG over Maternal Health KB (GAIN datasets + WHO/GHS guidelines in Twi/English)
  - Fine-tuned LLM (Llama-3 / Qwen / Mistral fine-tuned on health Q&A) for response generation in Twi
  - Symptom checker with disclaimers + escalation to human (SMS/telemedicine)
[Ecommerce Branch]
  - Product search / price check (voice query → vector search over catalog)
  - Cart / order via voice (integrate with local APIs like Jumia, Glovo, or market APIs)
  - Voice confirmation + payment (USSD/MoMo integration)
[General Chat]
  - Multilingual LLM with cultural grounding
       ↓
Response Generation (TTS in local dialect – Coqui XTTS or fine-tuned on GhanaNLP TTS data)
       ↓
Client Playback + Visual Transcript (with speaker labels)
```

**Key Technologies**:
- **Frontend**: Flutter (mobile-first, offline PWA) or React Native + WebSockets. Or simple web demo.
- **Backend**: FastAPI on Modal (ASGI for WebSockets).
- **ASR Core**: NVIDIA Parakeet (streaming multi-talker) or Whisper + streaming wrappers (WhisperLiveKit / WhisperStreaming). Fine-tune base on Ghana data via Modal.
- **Diarization + Voice ID**: NVIDIA Sortformer (streaming) + pyannote embeddings or ECAPA-TDNN for verification.
- **LLM**: Fine-tuned on GAIN datasets + RAG (LlamaIndex/Haystack). Serve with vLLM on Modal H100/A100.
- **TTS**: XTTS-v2 fine-tuned on GhanaNLP TTS corpora (or Piper for edge).
- **Data/Infra**: Modal Volumes for datasets/models; persistent storage; secrets for HF/Modal.
- **Privacy/Security**: Voice encryption in transit, on-device option for core ASR, anonymization, consent flows, Ghana DPA compliance.

**Latency Targets**:
- Partial transcription: <300ms
- Full utterance + response: <1.5s (target <800ms with caching/snapshots)
- Multi-speaker handling: Native in Parakeet pipeline.

### 2.2 Data Flow & Storage
- **Raw Audio**: Ephemeral (processed in streaming buffers; deleted post-transcription unless consented for improvement).
- **Transcripts + Metadata**: Stored encrypted (user_id hashed, speaker labels, timestamp, language confidence).
- **Health Records**: User-controlled (Voice ID links to profile). RAG knowledge base versioned.
- **Ecommerce**: Session-based or linked to user profile (MoMo integration).
- **Modal Volumes**: `/data/asr_models`, `/data/health_rag`, `/data/tts`.

---

## 3. Data Strategy & Datasets (Ready-to-Use)

**Priority Datasets (All public/HF)**:
1. **ASR Foundation (Ghana-specific)**:
   - UGSpeechData / University of Ghana: ~1000h per language (Akan/Twi, Ewe, Dagbani, etc.) + ~100h transcribed. Descriptions of culturally relevant images. Perfect for fine-tuning Whisper.
   - GhanaNLP Community (HF): navigation-corpus-twi-speech, ewe-speech, dagbani-speech, ga, UNICEF-Ghana-* -ASR, Bible audio-text for TTS.
   - AfriSpeech-200: Ghanaian accents (Twi etc.) + medical domain.

2. **Health-Specific**:
   - GAIN (Ghana AI Research Network) Maternal Health Q&A: 20k+ code-mixed Twi-English, Ga-English, etc. pairs. Topics: ANC, danger signs, nutrition, postpartum. **Gold for RAG + fine-tuning chat**.
   - Akan–English Maternal Health Parallel Corpus (Mendeley).
   - MedASR-Ghana: Pre-fine-tuned model for Ghanaian-accented medical English (start here for hybrid English + local).

3. **General / Parallel**:
   - GhanaNLP Parallel Corpora (Twi, Ewe, Ga, etc. – English aligned).
   - Twi Words Dataset, speech-text parallel corpora.

**Fine-Tuning Plan (Use Your Modal Credits)**:
- **ASR**: Start with `openai/whisper-small` or `large-v3-turbo`. Fine-tune with PEFT/LoRA on UGSpeechData + GhanaNLP (Twi priority). Use Modal's official `fine_tune_asr.py` example (H100, 3h timeout). Target WER <20-25% on held-out Ghana test sets.
- **Health LLM**: Fine-tune Llama-3-8B or Qwen2.5 on GAIN Q&A (LoRA). Or pure RAG first (cheaper/faster).
- **TTS**: Fine-tune XTTS on GhanaNLP TTS data for natural Twi prosody.
- **Parallel Experiments**: Launch multiple Modal jobs (different languages/configs) – your $200 covers several full runs.

**Data Pipeline**:
- Download via `datasets` lib.
- Preprocess: Normalize code-mixing, phonetic transcription where needed, augment with noise/speed for robustness.
- Splits: 80/10/10 train/val/test per language.
- Evaluation: WER (jiwer), CER, speaker diarization DER, end-to-end task success (health advice accuracy via human eval).

---

## 4. Core Components – Deep Technical Details

### 4.1 Real-Time Voice Pipeline (Modal Implementation)

Use/adapt Modal's **Parakeet Multi-talker** example (perfect for conversations with family/doctor present):

```python
# modal_parakeet_health_voice.py (core streaming service)
import modal
from dataclasses import dataclass
import torch
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.speaker_utils import SpeakerEmbeddings  # or pyannote

app = modal.App("ghana-health-voice")

image = modal.Image.debian_slim().pip_install(
    "nemo_toolkit[asr]", "torch", "transformers", "fastapi", "websockets"
)

@dataclass
class VoiceConfig:
    asr_model: str = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
    diar_model: str = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    language: str = "tw"  # Twi primary
    max_speakers: int = 4

@app.cls(gpu="H100", image=image, timeout=3600)
class GhanaVoiceTranscriber:
    @modal.enter()
    def setup(self):
        self.asr = ASRModel.from_pretrained(self.config.asr_model).eval().cuda()
        self.diar = ...  # Sortformer setup (streaming config)
        # Load fine-tuned Ghana adapter if available
        self.processor = ...  # Whisper processor or NeMo equivalent for lang ID

    @modal.batched(max_batch_size=1)  # or streaming handler
    def transcribe_stream(self, audio_chunk: bytes, speaker_id: str = None):
        # Process chunk → features
        # Run ASR + Diarization jointly (Parakeet native multi-talker)
        transcription, speakers = self.asr.transcribe(audio_chunk, ...)
        # Voice ID: If enrolled, verify embedding
        if speaker_id:
            verified = self.verify_speaker(embedding, speaker_id)
        return {"text": transcription, "speaker": speakers[0], "verified": verified, "lang": "tw"}

    def verify_speaker(self, embedding, enrolled_id):
        # Cosine similarity on embeddings (store enrolled in Volume or DB)
        sim = torch.cosine_similarity(...)
        return sim > 0.7  # threshold

# WebSocket endpoint for real-time
@app.function()
@modal.asgi_app()
def voice_ws():
    from fastapi import FastAPI, WebSocket
    web_app = FastAPI()
    @web_app.websocket("/ws/voice")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        while True:
            audio = await websocket.receive_bytes()
            result = GhanaVoiceTranscriber().transcribe_stream.remote(audio)
            await websocket.send_json(result)
    return web_app
```

**Deployment**: `modal deploy modal_parakeet_health_voice.py`

For lighter/edge: WhisperLiveKit or faster-whisper + Diart (real-time diarization).

**Voice ID Flow**:
1. Enrollment: User records 30-60s passphrase in Twi → extract embedding → store hashed.
2. Verification: On each session or sensitive query (health records), re-verify.
3. Multi-user households: Diarization separates speakers; verification personalizes responses.

### 4.2 Health Intelligence Layer

**RAG + LLM**:
- **Knowledge Base**: GAIN 20k Q&A (Twi-English code-mixed) + WHO/Ghana Health Service guidelines translated/aligned. Embed with multilingual model (e.g., paraphrase-multilingual-MiniLM).
- **Retrieval**: Hybrid (BM25 + vector) + reranker.
- **Generation**: Fine-tuned model or strong base (Qwen2.5-7B or Llama-3.1-8B) with system prompt for cultural sensitivity, disclaimers ("This is not medical advice – consult professional"), and escalation.
- **Fine-tuning on Modal**: Use Unsloth or Axolotl example for efficiency. Train on GAIN + synthetic health dialogues.

**Symptom Checker**: Rule-based + LLM (or small classifier) for common issues (fever, pregnancy symptoms). Output in Twi with severity flags.

**Safety**: Heavy disclaimers, human escalation path (SMS to CHW or hotline), bias auditing on Ghana data.

### 4.3 Ecommerce Module

- **Intent**: Separate classifier (or LLM tool-calling) for "shopping", "price check", "order".
- **Catalog**: Vector DB of products (name, description, price in Twi/English, category – local staples, OTC meds, household).
- **Flow**: Voice query → ASR → Intent + Entity extraction (product, quantity) → Search → Voice confirm + add to cart → MoMo/USSD payment.
- **Integration**: APIs for local platforms or mock catalog initially. Voice commerce huge for accessibility.

Example intent prompt for LLM:
```
You are a helpful Ghanaian market assistant speaking in Twi-English mix.
User said: [transcribed text]
If shopping intent, extract product and quantity. Respond naturally in mix.
```

### 4.4 TTS & Multimodal Output

- Fine-tune Coqui XTTS-v2 or use Piper (fast edge) on GhanaNLP TTS data for natural prosody/accents.
- Output: Text + audio + visual transcript (speaker-labeled chat UI).

---

## 5. Implementation Roadmap (Phased for Grok Build)

**Phase 0: Foundation (1-2 weeks)**
- Set up Modal workspace + secrets (HF, W&B).
- Download & explore key datasets (UGSpeechData Twi subset, GAIN Maternal Q&A).
- Baseline: Run Modal Whisper fine-tune example on Twi data. Measure WER.
- Deploy basic streaming transcription demo (adapt Parakeet example).

**Phase 1: MVP Voice Health Chat – Twi (4-6 weeks)**
- Fine-tune ASR on Ghana data (Modal parallel jobs).
- Build RAG over GAIN dataset + basic medical KB.
- Integrate streaming pipeline (ASR + simple diarization).
- Simple LLM chat (RAG + generation in Twi mix).
- Mobile/web frontend with mic + transcript.
- Basic Voice ID (enrollment + verification stub).
- Test with real Ghanaian speakers (WER, usability).

**Phase 2: Full Real-time + Multi-speaker + Ecommerce (4-6 weeks)**
- Integrate NVIDIA Parakeet + Sortformer for true real-time multi-talker.
- Add full Voice ID + personalization.
- Ecommerce intent + catalog integration (voice search/order).
- TTS in Twi.
- Offline edge version (ONNX export for mobile).
- Dashboard for monitoring (Modal + custom).

**Phase 3: Scale & Polish (Ongoing)**
- Add more languages (Ewe, Ga, Dagbani) via parallel fine-tunes.
- Clinical validation + partnerships (CHW, Ministry of Health pilots).
- Advanced features: Emotion detection (mental health), image upload for symptoms (multimodal), offline-first PWA.
- Monetization: Freemium (basic health free; premium ecommerce/ personalized), B2B for clinics.
- Evaluation: Automated (WER, task success) + human (cultural appropriateness, safety).

**Total MVP Timeline**: 8-12 weeks to production-ready Twi health voice companion.

---

## 6. Risks, Mitigations & Best Practices

- **Data Scarcity/Quality**: Use existing large corpora + augmentation. Human validation loops.
- **Code-mixing & Dialects**: Train on authentic code-mixed data (GAIN). Multi-dialect fine-tuning.
- **Accuracy in Health**: Strong disclaimers + "not a doctor" + escalation. Audit for bias (gender, region).
- **Privacy**: Voice data sensitive – minimize storage, encrypt, on-device core, consent UI. Ghana DPA + GDPR alignment.
- **Latency/Cost**: Modal snapshots for fast cold starts. Batch where possible. Monitor credits.
- **Adoption**: Co-design with users (women's groups, CHWs). Offline mode critical for rural.
- **Technical**: Streaming stability – test with real noisy environments (market, home). Fallback to batch.
- **Ethical**: Inclusive (accents, impairments – see UGAkan-ImpairedSpeechData). No over-promising on medical advice.

---

## 7. Code & Deployment Snippets

(See sections above for core Modal classes.)

**Quick Start Commands**:
```bash
# 1. Modal setup (user has credits)
pip install modal
modal token new
modal secret create huggingface-secret HF_TOKEN=...

# 2. Clone examples & adapt
git clone https://github.com/modal-labs/modal-examples
cd modal-examples/06_gpu_and_ml/openai_whisper
# Edit fine_tune_asr.py for Twi + your datasets

modal run fine_tune_asr.py  # baseline
# Then deploy your custom parakeet_health_voice.py

# 3. Frontend prototype
# Use Streamlit or simple HTML/JS WebSocket client for demo
```

**HF Model Upload Example** (after fine-tune):
```python
model.push_to_hub("yourusername/whisper-twi-health-v1")
```

---

## 8. Resources & References

**Datasets**:
- UGSpeechData (University of Ghana) – primary ASR corpus.
- GhanaNLP HF org – speech/TTS/ASR.
- GAIN Maternal Health Q&A (Kaggle) – health chat goldmine.
- AfriSpeech-200, MedASR-Ghana.

**Modal Examples** (directly usable):
- Fine-tune Whisper ASR.
- Parakeet Multi-talker streaming + Sortformer diarization.
- Unsloth LLM fine-tuning.

**Tools**:
- NeMo (NVIDIA) for Parakeet/Sortformer.
- Transformers + PEFT for fine-tuning.
- LlamaIndex / Haystack for RAG.
- Flutter/React Native for app.

**Next for Builder**: Start with Phase 0 baseline on Modal today. Share progress/logs for iteration. Use the Grok Offers in Section 9 for rapid progress.

**Call to Action**: This is production-viable with your Modal credits + public datasets. Let's build something that truly serves Ghanaians – voice in their language, for their health and livelihoods.

**Contact/Iteration**: Provide feedback or claim an Offer (Section 9) and I'll deliver immediately.

---

## 9. Grok's Specific Offers for Rapid Implementation

**I can immediately generate/deliver the following on request** (leveraging your Modal credits and the datasets above). These will accelerate your portfolio piece significantly, especially pushing past 34% WER with punctuation, POS, and RLHF/DPO:

### Offer 1: Post-Processing Pipeline (Punctuation + POS + Entities)
- Ready-to-run Python module with DeepPunct-style restoration, Stanza/POS tagging (Twi-adapted), and medical entity extraction from GAIN datasets.
- Integrates directly after ASR output for instant readability/intent boost.

### Offer 2: Next Fine-Tuning Script for Modal
- Customized `modal_asr_twi_health.py` extending the official Whisper example.
- Includes LoRA, health-domain data mix (GAIN + UGSpeechData), hyperparameter sweeps, and W&B logging.
- Commands to launch parallel experiments targeting <25% WER.

### Offer 3: DPO/RLHF Setup for Preference Tuning
- Preference data generator (from multiple ASR outputs + LLM judge or human labels).
- Full Axolotl/Unsloth DPO config + Modal training job for "human-preferred" transcripts (better punctuation, medical accuracy, natural Twi).
- This is portfolio gold — demonstrates advanced alignment.

### Offer 4: End-to-End Streaming Voice Demo
- Full Modal WebSocket service (Parakeet/Sortformer + Voice ID + Health RAG chat).
- Simple frontend (HTML/JS or Flutter stub) for live Twi voice → punctuated transcript → response.
- Ecommerce intent stub included.

### Offer 5: Complete MVP Repo Structure
- Git-ready folder layout with all scripts, Docker/Modal configs, evaluation notebooks, and deployment instructions.

**How to Claim Any Offer**: Just say e.g. “Generate Offer 2 – Twi fine-tuning script” or “Deliver the full post-processing pipeline + DPO setup”.

These are production-ready starting points that will take your 34% WER model to a compelling, real-time Health AI demo.

---

*This document is comprehensive yet actionable. Copy-paste sections into tickets, prompts for code gen, or architecture reviews. Ready to build.* 

**End of Spec**