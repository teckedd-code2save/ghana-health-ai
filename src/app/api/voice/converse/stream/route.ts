import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { isModalAsrConfigured, modalTranscribe } from "@/lib/modal-asr";
import { clientIp, rateLimit } from "@/lib/rate-limit";
import { runConversationTurn } from "@/lib/conversation-turn";
import type { LanguageCode } from "@prisma/client";
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

function streamHeaders() {
  return {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
  };
}

function formatEvent(event: string, data: unknown) {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function publicVoicePayload(input: {
  turn: Awaited<ReturnType<typeof runConversationTurn>>;
  asr: Awaited<ReturnType<typeof modalTranscribe>>;
  started: number;
}) {
  const { turn, asr, started } = input;
  return {
    conversationId: turn.conversationId,
    asr: {
      text: asr.text,
      model: asr.model,
      latencyMs: asr.latency_ms,
      language: asr.language,
      languageProbability: asr.language_probability,
      duration: asr.duration,
      rms: asr.rms,
      route: asr.route?.name,
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
  };
}

export async function POST(req: Request) {
  const encoder = new TextEncoder();
  const started = Date.now();

  return new Response(
    new ReadableStream({
      async start(controller) {
        let closed = false;
        const send = (event: string, data: unknown) => {
          if (closed) return false;
          try {
            controller.enqueue(encoder.encode(formatEvent(event, data)));
            return true;
          } catch (error) {
            if (
              error instanceof Error &&
              (error.name === "TypeError" || error.message.includes("Controller is already closed"))
            ) {
              closed = true;
              return false;
            }
            throw error;
          }
        };

        try {
          const ip = clientIp(req);
          const rl = rateLimit(`converse:stream:${ip}`, 12, 60);
          if (!rl.allowed) {
            send("error", { error: "Too many voice turns — wait a minute", status: 429 });
            return;
          }

          if (!isModalAsrConfigured()) {
            send("error", {
              error: "MODAL_ASR_URL not configured — real ASR required",
              status: 503,
            });
            return;
          }

          const user = await getSessionUser();
          const form = await req.formData();
          const file = form.get("audio");
          const lang = form.get("language");
          const cid = form.get("conversationId");
          const sp = form.get("speak");
          const focusValue = form.get("focus");
          const instructionValue = form.get("instruction");
          const asrModelValue = form.get("asrModel");
          const understandingModelValue = form.get("understandingModelMode");

          const language: LanguageCode =
            typeof lang === "string" && ["tw", "en", "ga"].includes(lang)
              ? (lang as LanguageCode)
              : user?.preferredLang ?? "tw";
          const conversationId = typeof cid === "string" && cid ? cid : undefined;
          const speak = !(sp === "false" || sp === "0");
          const focus =
            focusValue === "commerce" || focusValue === "health" ? focusValue : undefined;
          const instruction =
            typeof instructionValue === "string" && instructionValue.trim()
              ? instructionValue.slice(0, 240)
              : undefined;
          const asrModel =
            typeof asrModelValue === "string" && asrModelValue
              ? asrModelValue
              : undefined;
          const understandingModelMode =
            understandingModelValue === "shadow" || understandingModelValue === "assist"
              ? understandingModelValue
              : undefined;

          if (!(file instanceof Blob)) {
            send("error", { error: "Audio required for real conversation", status: 400 });
            return;
          }

          const audio = await file.arrayBuffer();
          if (audio.byteLength < 100) {
            send("error", { error: "Audio required for real conversation", status: 400 });
            return;
          }

          send("stage", { name: "accepted", at: Date.now() });
          send("stage", { name: "asr_started", at: Date.now() });
          const asr = await modalTranscribe(audio, {
            language: language === "en" ? "en" : undefined,
            contentType: file.type || "audio/webm",
            filename: "utterance.webm",
            asrModel,
          });
          if (asr.error || !asr.text?.trim()) {
            console.error("[voice/converse/stream:asr]", {
              error: asr.error || "Empty transcription",
              model: asr.model,
              route: asr.route?.name,
            });
            send("error", { error: unclearRecording.message, status: unclearRecording.status });
            return;
          }
          send("asr", {
            text: asr.text,
            model: asr.model,
            latencyMs: asr.latency_ms,
            language: asr.language,
            languageProbability: asr.language_probability,
            duration: asr.duration,
            rms: asr.rms,
            route: asr.route?.name,
          });
          send("stage", { name: "asr_final", detail: asr.model, at: Date.now() });

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
            onStage: (stage) => {
              if (stage.name === "reply_delta") {
                send("reply_delta", { chunk: stage.chunk, at: stage.at });
                return;
              }
              send("stage", stage);
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

          send("final", publicVoicePayload({ turn, asr, started }));
        } catch (e) {
          console.error("[voice/converse/stream]", e);
          const failure = publicFailure(e, "voice");
          send("error", { error: failure.message, status: failure.status });
        } finally {
          if (!closed) {
            closed = true;
            try {
              controller.close();
            } catch (error) {
              if (
                !(
                  error instanceof Error &&
                  (error.name === "TypeError" ||
                    error.message.includes("Controller is already closed"))
                )
              ) {
                throw error;
              }
            }
          }
        }
      },
    }),
    { headers: streamHeaders() },
  );
}
