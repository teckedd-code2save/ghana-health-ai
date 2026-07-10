import { VoicePanel } from "@/components/voice-panel";

export default function VoicePage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-3xl">Voice pipeline</h1>
        <p className="text-sm text-[var(--fg-muted)]">
          Modal Whisper Round 2 ASR → LLM → Akan TTS, plus audio Voice ID enroll/verify.
        </p>
      </div>
      <VoicePanel />
    </div>
  );
}
