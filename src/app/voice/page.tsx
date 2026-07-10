import { VoicePanel } from "@/components/voice-panel";

export default function VoicePage() {
  return (
    <div className="space-y-2">
      <p className="text-center text-sm text-[var(--fg-muted)]">
        Speak · we listen · you confirm · we answer
      </p>
      <VoicePanel />
    </div>
  );
}
