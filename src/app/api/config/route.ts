import { jsonOk } from "@/lib/api";
import { isPaystackConfigured, getPaystackPublicKey } from "@/lib/paystack";
import { isLlmConfigured } from "@/lib/llm";
import { isModalAsrConfigured } from "@/lib/modal-asr";
import { isModalTtsConfigured } from "@/lib/modal-tts";

/** Public runtime flags for the client (no secrets). */
export async function GET() {
  return jsonOk({
    voiceMode: process.env.VOICE_MODE || "modal",
    modalAsr: isModalAsrConfigured(),
    modalTts: isModalTtsConfigured(),
    llm: isLlmConfigured(),
    paystack: isPaystackConfigured(),
    paystackPublicKey: getPaystackPublicKey() || process.env.NEXT_PUBLIC_PAYSTACK_PUBLIC_KEY || "",
    asrModel: "teckedd/whisper-small-waxal-round2-specaug-v1",
    ttsModel: "facebook/mms-tts-aka | facebook/mms-tts-eng",
    languages: ["tw", "en", "ga", "ee", "dag"],
    hotline: process.env.HEALTH_ESCALATION_HOTLINE || "112 / community health worker",
    appUrl: process.env.NEXT_PUBLIC_APP_URL,
  });
}
