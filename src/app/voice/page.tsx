import { VoicePanel } from "@/components/voice-panel";

export default function VoicePage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-3xl">Voice pipeline</h1>
        <p className="text-sm text-[var(--fg-muted)]">
          ASR stub + Voice ID enrollment. Modal Parakeet scripts live under <code>/modal</code>.
        </p>
      </div>
      <VoicePanel />
    </div>
  );
}
