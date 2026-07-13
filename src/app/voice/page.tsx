import { VoicePanel } from "@/components/voice-panel";

export default function VoicePage() {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="display text-2xl md:text-3xl">Speak</h1>
        <p className="mt-1.5 text-sm text-[var(--fg-muted)]">
          Say what’s on your mind. Confirm, then hear the reply.
        </p>
      </div>
      <VoicePanel />
    </div>
  );
}
