import { ChatPanel } from "@/components/chat-panel";

export default function ChatPage() {
  return (
    <div className="chat-page">
      <div className="text-center md:text-left">
        <h1 className="display text-2xl md:text-3xl">Chat</h1>
        <p className="mt-1.5 text-sm text-[var(--fg-muted)]">
          Voice or text — your conversation stays here.
        </p>
      </div>
      <ChatPanel />
    </div>
  );
}
