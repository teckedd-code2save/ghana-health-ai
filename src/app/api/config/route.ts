import { jsonOk } from "@/lib/api";
import { isPaystackConfigured, getPaystackPublicKey } from "@/lib/paystack";
import { isLlmConfigured } from "@/lib/llm";
import { isModalAsrConfigured } from "@/lib/modal-asr";
import { isModalTtsConfigured } from "@/lib/modal-tts";
import { isEmbedConfigured } from "@/lib/embed";

/** Public runtime flags (no secrets). */
export async function GET() {
  return jsonOk({
    voiceMode: process.env.VOICE_MODE || "modal",
    modalAsr: isModalAsrConfigured(),
    modalTts: isModalTtsConfigured(),
    abenaEmbed: isEmbedConfigured(),
    llm: isLlmConfigured(),
    paystack: isPaystackConfigured(),
    paystackPublicKey: getPaystackPublicKey() || process.env.NEXT_PUBLIC_PAYSTACK_PUBLIC_KEY || "",
    stack: {
      asr: "teckedd/gha-whisper-small-twi-v6 (Waxal multi-domain train)",
      embed: "Ghana-NLP/abena-base-asante-twi-uncased",
      tts: "facebook/mms-tts-aka",
      language: "Twi default; English only if preferred",
      docs: "docs/research-stack.md",
    },
    languages: ["tw", "en", "ga", "ee", "dag"],
    hotline: process.env.HEALTH_ESCALATION_HOTLINE || "112 / community health worker",
    appUrl: process.env.NEXT_PUBLIC_APP_URL,
  });
}
