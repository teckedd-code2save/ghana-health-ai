"use client";

import { useCallback, useEffect, useState } from "react";
import { MessageSquare, MoreHorizontal, Plus, Trash2 } from "lucide-react";
import {
  CONVERSATION_EVENT,
  forgetConversation,
  registeredConversationIds,
  selectConversation,
  startNewConversation,
  type ConversationChange,
} from "@/lib/conversation-store";

type ConversationSummary = {
  id: string;
  title: string | null;
  updatedAt: string;
  _count: { messages: number };
};

export function SessionDrawer({ onNavigate }: { onNavigate: () => void }) {
  const [sessions, setSessions] = useState<ConversationSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [menuId, setMenuId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ConversationSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const ids = registeredConversationIds();
    const query = ids.length ? `?ids=${encodeURIComponent(ids.join(","))}` : "";
    try {
      const response = await fetch(`/api/chat${query}`, { cache: "no-store" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not load chats");
      setSessions(data.conversations ?? []);
    } catch {
      setSessions([]);
    }
  }, []);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => void load(), 0);
    const onChange = (event: Event) => {
      const detail = (event as CustomEvent<ConversationChange>).detail;
      if (detail?.type === "refresh" || detail?.type === "delete") void load();
    };
    window.addEventListener(CONVERSATION_EVENT, onChange);
    return () => {
      window.clearTimeout(initialLoad);
      window.removeEventListener(CONVERSATION_EVENT, onChange);
    };
  }, [load]);

  function createNew() {
    startNewConversation();
    onNavigate();
    if (window.location.pathname !== "/chat") window.location.assign("/chat");
  }

  function openSession(conversationId: string) {
    selectConversation(conversationId);
    onNavigate();
    if (window.location.pathname !== "/chat") window.location.assign("/chat");
  }

  async function deleteSession() {
    if (!deleteTarget || busy) return;
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/chat?conversationId=${encodeURIComponent(deleteTarget.id)}`,
        { method: "DELETE" },
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not delete chat");
      forgetConversation(deleteTarget.id);
      window.dispatchEvent(
        new CustomEvent<ConversationChange>(CONVERSATION_EVENT, {
          detail: { type: "delete", conversationId: deleteTarget.id },
        }),
      );
      setSessions((current) => current.filter((item) => item.id !== deleteTarget.id));
      setDeleteTarget(null);
      setMenuId(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not delete chat");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="session-drawer">
      <button type="button" className="session-new" onClick={createNew}>
        <Plus className="h-4 w-4" />
        <span>New chat</span>
      </button>

      <div className="session-heading">Recent chats</div>
      <div className="session-list">
        {sessions.length === 0 ? (
          <p className="session-empty">Your conversations will appear here.</p>
        ) : (
          sessions.map((session) => (
            <div className="session-row" key={session.id}>
              <button
                type="button"
                className="session-open"
                onClick={() => openSession(session.id)}
              >
                <MessageSquare className="h-4 w-4 shrink-0" />
                <span>
                  <strong>{session.title || "Untitled conversation"}</strong>
                  <small>{session._count.messages} messages</small>
                </span>
              </button>
              <button
                type="button"
                className="session-more"
                aria-label={`Actions for ${session.title || "conversation"}`}
                aria-expanded={menuId === session.id}
                onClick={() => setMenuId((current) => (current === session.id ? null : session.id))}
              >
                <MoreHorizontal className="h-4 w-4" />
              </button>
              {menuId === session.id && (
                <div className="session-context-menu">
                  <button type="button" onClick={() => setDeleteTarget(session)}>
                    <Trash2 className="h-4 w-4" /> Delete
                  </button>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {deleteTarget && (
        <div className="session-confirm-backdrop" role="presentation">
          <div className="session-confirm" role="alertdialog" aria-modal="true" aria-labelledby="delete-chat-title">
            <h2 id="delete-chat-title">Delete this chat?</h2>
            <p>This removes the conversation and its messages. This cannot be undone.</p>
            {error && <p className="session-confirm-error">{error}</p>}
            <div>
              <button type="button" className="session-cancel" onClick={() => setDeleteTarget(null)} disabled={busy}>
                Cancel
              </button>
              <button type="button" className="session-delete" onClick={() => void deleteSession()} disabled={busy}>
                {busy ? "Deleting…" : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
