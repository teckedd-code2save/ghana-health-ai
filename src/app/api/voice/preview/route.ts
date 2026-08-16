import { getSessionUser } from "@/lib/auth";
import { jsonError, jsonOk } from "@/lib/api";
import { isModalAsrConfigured, modalTranscribe } from "@/lib/modal-asr";
import { clientIp, rateLimit } from "@/lib/rate-limit";

export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`voice:preview:${ip}`, 8, 60);
    if (!rl.allowed) return jsonError("Too many live previews", 429);

    if (!isModalAsrConfigured()) {
      return jsonError("MODAL_ASR_URL not configured — real ASR required", 503);
    }

    const user = await getSessionUser();
    const contentType = req.headers.get("content-type") ?? "";
    if (!contentType.includes("multipart/form-data")) {
      return jsonError("Audio form required", 400);
    }

    const form = await req.formData();
    const file = form.get("audio");
    const lang = form.get("language");
    const asrModelValue = form.get("asrModel");
    if (!(file instanceof Blob)) return jsonError("Audio required", 400);

    const audio = await file.arrayBuffer();
    if (audio.byteLength < 800) return jsonError("Audio too short", 400);

    const language = typeof lang === "string" ? lang : user?.preferredLang;
    const result = await modalTranscribe(audio, {
      contentType: file.type || "audio/webm",
      filename: "preview.webm",
      language: language === "en" ? "en" : undefined,
      asrModel:
        typeof asrModelValue === "string" && asrModelValue ? asrModelValue : undefined,
    });

    if (result.error && !result.text?.trim()) {
      return jsonOk({
        preview: {
          text: "",
          provisional: true,
          error: result.error,
          model: result.model,
          latencyMs: result.latency_ms,
          language: result.language,
          languageProbability: result.language_probability,
          route: result.route?.name,
          duration: result.duration,
          rms: result.rms,
        },
      });
    }

    return jsonOk({
      preview: {
        text: result.text.trim(),
        provisional: true,
        language: result.language === "en" ? "en" : "tw",
        languageProbability: result.language_probability,
        route: result.route?.name,
        duration: result.duration,
        rms: result.rms,
        model: result.model,
        latencyMs: result.latency_ms,
      },
    });
  } catch (e) {
    console.error("[voice/preview]", e);
    return jsonError(e instanceof Error ? e.message : "Voice preview failed", 500);
  }
}
