import { z } from "zod";
import { isModalTtsConfigured, modalSpeak, type TtsProvider } from "@/lib/modal-tts";
import { jsonError, jsonOk } from "@/lib/api";
import { clientIp, rateLimit } from "@/lib/rate-limit";

const schema = z.object({
  text: z.string().min(1).max(2000),
  language: z.enum(["tw", "en", "ga"]).optional(),
  provider: z.enum(["mms", "stable-twi", "nano-twi", "qwen"]).optional(),
});

export async function POST(req: Request) {
  try {
    const ip = clientIp(req);
    const rl = rateLimit(`tts:${ip}`, 20, 60);
    if (!rl.allowed) return jsonError("Too many TTS requests", 429);

    if (!isModalTtsConfigured()) {
      return jsonError("MODAL_TTS_URL not configured", 503);
    }

    const body = schema.parse(await req.json());
    const result = await modalSpeak(body.text, body.language ?? "tw", {
      provider: body.provider as TtsProvider | undefined,
    });
    if (result.error || !result.audio_base64) {
      return jsonError(result.error || "TTS failed", 502);
    }

    return jsonOk({
      audioBase64: result.audio_base64,
      sampleRate: result.sample_rate,
      format: result.format,
      model: result.model,
      provider: result.provider,
      latencyMs: result.latency_ms,
      duration: result.duration,
    });
  } catch (e) {
    if (e instanceof z.ZodError) return jsonError(e.issues[0]?.message ?? "Invalid input");
    console.error(e);
    return jsonError("TTS failed", 500);
  }
}
