import { ChatPanel } from "@/components/chat-panel";

export default function ChatPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-3xl">Companion chat</h1>
        <p className="text-sm text-[var(--fg-muted)]">
          LLM understanding with real mic → Twi ASR. Optional Akan TTS on replies.
        </p>
      </div>
      <ChatPanel />
    </div>
  );
}
