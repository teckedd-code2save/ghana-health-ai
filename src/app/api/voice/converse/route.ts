import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { isModalAsrConfigured, modalTranscribe } from "@/lib/modal-asr";
import { jsonError, jsonOk } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import type { LanguageCode } from "@prisma/client";
import { runConversationTurn } from "@/lib/conversation-turn";
import { publicFailure, unclearRecording } from "@/lib/public-errors";

function languageFromAsr(
  asr: Awaited<ReturnType<typeof modalTranscribe>>,
  fallback: LanguageCode,
): LanguageCode {
  const confidence = asr.language_probability;
  if (typeof confidence === "number" && confidence > 0 && confidence < 0.55) {
    return fallback;
  }

  const language = asr.language?.trim().toLowerCase().split(/[-_]/)[0] ?? "";
  if (["en", "eng", "english"].includes(language)) return "en";
  if (["tw", "twi", "ak", "aka", "akan"].includes(language)) return "tw";
  if (["ga", "gaa"].includes(language)) return "ga";
  return fallback;
}

/**
 * Real voice loop:
 *   audio → Modal ASR (Twi Whisper) → LLM understand → Modal TTS (Akan MMS)
 */
export async function POST(req: Request) {
  const started = Date.now();
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`converse:${ip}`, 12, 60);
    if (!rl.allowed) return jsonError("Too many voice turns — wait a minute", 429);

    if (!isModalAsrConfigured()) {
      return jsonError("MODAL_ASR_URL not configured — real ASR required", 503);
    }

    const user = await getSessionUser();
    const contentType = req.headers.get("content-type") ?? "";
    let audio: ArrayBuffer | null = null;
    let language: LanguageCode = user?.preferredLang ?? "tw";
    let conversationId: string | undefined;
    let speak = true;
    let focus: "health" | "commerce" | undefined;
    let instruction: string | undefined;
    let asrModel: string | undefined;
    let understandingModelMode: "shadow" | "assist" | undefined;

    if (contentType.includes("multipart/form-data")) {
      const form = await req.formData();
      const file = form.get("audio");
      if (file instanceof Blob) audio = await file.arrayBuffer();
      const lang = form.get("language");
      if (typeof lang === "string" && ["tw", "en", "ga"].includes(lang)) {
        language = lang as LanguageCode;
      }
      const cid = form.get("conversationId");
      if (typeof cid === "string" && cid) conversationId = cid;
      const sp = form.get("speak");
      if (sp === "false" || sp === "0") speak = false;
      const model = form.get("asrModel");
      if (typeof model === "string" && model) asrModel = model;
      const focusValue = form.get("focus");
      if (focusValue === "health" || focusValue === "commerce") focus = focusValue;
      const instructionValue = form.get("instruction");
      if (typeof instructionValue === "string" && instructionValue.trim()) {
        instruction = instructionValue.slice(0, 240);
      }
      const understandingValue = form.get("understandingModelMode");
      if (understandingValue === "shadow" || understandingValue === "assist") {
        understandingModelMode = understandingValue;
      }
    } else {
      audio = await req.arrayBuffer();
    }

    if (!audio || audio.byteLength < 100) {
      return jsonError("Audio required for real conversation", 400);
    }

    // 1) ASR
    const asr = await modalTranscribe(audio, {
      language: language === "en" ? "en" : undefined,
      contentType: "audio/webm",
      asrModel,
    });
    if (asr.error || !asr.text?.trim()) {
      console.error("[voice/converse:asr]", {
        error: asr.error || "Empty transcription",
        model: asr.model,
        route: asr.route?.name,
      });
      return jsonError(unclearRecording.message, unclearRecording.status);
    }
    const replyLanguage = languageFromAsr(asr, language);

    const turn = await runConversationTurn({
      user,
      ip,
      text: asr.text,
      language: replyLanguage,
      focus,
      instruction,
      conversationId,
      channel: "VOICE",
      speak,
      understandingModelMode,
      transcript: {
        mode: "asr",
        model: asr.model,
        latencyMs: asr.latency_ms,
        speaker: asr.speaker,
        language: asr.language,
        languageProbability: asr.language_probability,
        route: asr.route?.name,
        duration: asr.duration,
        rms: asr.rms,
      },
    });

    await prisma.transcriptSession.create({
      data: {
        userId: user?.id,
        language: replyLanguage,
        rawText: asr.text,
        punctuated: asr.text,
        speakers: [{ label: asr.speaker, model: asr.model }],
        audioDeleted: true,
        retained: false,
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
      },
    });

    return jsonOk({
      conversationId: turn.conversationId,
      asr: {
        text: asr.text,
        model: asr.model,
        latencyMs: asr.latency_ms,
        language: asr.language,
        languageProbability: asr.language_probability,
        route: asr.route?.name,
        duration: asr.duration,
        rms: asr.rms,
      },
      understanding: {
        reply: turn.reply,
        intent: turn.understanding.intent,
        severity: turn.understanding.severity,
        escalate: turn.understanding.escalate,
        engine: turn.understanding.engine,
        health: turn.understanding.health ?? null,
        commerce: turn.understanding.commerce ?? null,
        commerceExecution: turn.commerceExecution ?? null,
        review: turn.understanding.review ?? null,
        synthesis: turn.understanding.synthesis ?? null,
        comprehension: turn.understanding.comprehension ?? null,
      },
      userMessage: {
        id: turn.userMessageId,
        content: asr.text,
      },
      message: {
        id: turn.assistantId,
        content: turn.reply,
      },
      tts: turn.tts,
      stage: turn.stage,
      totalLatencyMs: Date.now() - started,
    });
  } catch (e) {
    console.error("[voice/converse]", e);
    const failure = publicFailure(e, "voice");
    return jsonError(failure.message, failure.status);
  }
}
