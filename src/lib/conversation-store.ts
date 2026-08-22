export const CONVERSATION_EVENT = "gha:conversation-change";
const REGISTRY_KEY = "gha:conversation-registry";
const ACTIVE_KEY = "gha:active-conversation";

export type ConversationChange =
  | { type: "new" }
  | { type: "select"; conversationId: string }
  | { type: "delete"; conversationId: string }
  | { type: "refresh" };

export function registeredConversationIds() {
  if (typeof window === "undefined") return [];
  try {
    const value = JSON.parse(window.localStorage.getItem(REGISTRY_KEY) || "[]");
    return Array.isArray(value)
      ? value.filter((id): id is string => typeof id === "string").slice(0, 50)
      : [];
  } catch {
    return [];
  }
}

export function registerConversation(conversationId?: string) {
  if (!conversationId || typeof window === "undefined") return;
  const ids = registeredConversationIds();
  window.localStorage.setItem(
    REGISTRY_KEY,
    JSON.stringify([conversationId, ...ids.filter((id) => id !== conversationId)].slice(0, 50)),
  );
  window.localStorage.setItem(ACTIVE_KEY, conversationId);
  emitConversationChange({ type: "refresh" });
}

export function forgetConversation(conversationId: string) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(
    REGISTRY_KEY,
    JSON.stringify(registeredConversationIds().filter((id) => id !== conversationId)),
  );
  if (window.localStorage.getItem(ACTIVE_KEY) === conversationId) {
    window.localStorage.removeItem(ACTIVE_KEY);
  }
  for (const key of Object.keys(window.localStorage)) {
    if (key.startsWith("gha:home-voice:") && window.localStorage.getItem(key) === conversationId) {
      window.localStorage.removeItem(key);
    }
  }
}

export function startNewConversation() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(ACTIVE_KEY);
  for (const key of Object.keys(window.localStorage)) {
    if (key.startsWith("gha:home-voice:")) window.localStorage.removeItem(key);
  }
  emitConversationChange({ type: "new" });
}

export function selectConversation(conversationId: string) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(ACTIVE_KEY, conversationId);
  emitConversationChange({ type: "select", conversationId });
}

export function emitConversationChange(detail: ConversationChange) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent<ConversationChange>(CONVERSATION_EVENT, { detail }));
}
