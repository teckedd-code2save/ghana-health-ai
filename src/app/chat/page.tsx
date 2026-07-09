import { ChatPanel } from "@/components/chat-panel";

export default function ChatPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-3xl">Companion chat</h1>
        <p className="text-sm text-[var(--fg-muted)]">
          Health RAG, ecommerce intent, and voice stub input — all from live API routes.
        </p>
      </div>
      <ChatPanel />
    </div>
  );
}
