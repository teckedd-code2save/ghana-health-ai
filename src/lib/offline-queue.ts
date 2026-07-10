/**
 * Client-side offline queue (localStorage).
 * Replays chat/cart actions when back online.
 */

export type QueuedAction = {
  id: string;
  kind: "chat" | "cart";
  payload: Record<string, unknown>;
  createdAt: string;
};

const KEY = "gha_offline_queue_v1";

export function loadQueue(): QueuedAction[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]") as QueuedAction[];
  } catch {
    return [];
  }
}

function saveQueue(items: QueuedAction[]) {
  localStorage.setItem(KEY, JSON.stringify(items.slice(-50)));
}

export function enqueueOffline(kind: QueuedAction["kind"], payload: Record<string, unknown>) {
  const items = loadQueue();
  items.push({
    id: crypto.randomUUID(),
    kind,
    payload,
    createdAt: new Date().toISOString(),
  });
  saveQueue(items);
  return items.length;
}

export async function flushOfflineQueue(): Promise<{ flushed: number; failed: number }> {
  const items = loadQueue();
  if (!items.length) return { flushed: 0, failed: 0 };
  const remaining: QueuedAction[] = [];
  let flushed = 0;
  let failed = 0;

  for (const item of items) {
    try {
      if (item.kind === "chat") {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(item.payload),
        });
        if (!res.ok) throw new Error("chat failed");
      } else if (item.kind === "cart") {
        const res = await fetch("/api/cart", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(item.payload),
        });
        if (!res.ok) throw new Error("cart failed");
      }
      flushed += 1;
    } catch {
      remaining.push(item);
      failed += 1;
    }
  }
  saveQueue(remaining);
  return { flushed, failed };
}

export function setupOfflineFlush() {
  if (typeof window === "undefined") return;
  window.addEventListener("online", () => {
    void flushOfflineQueue();
  });
  if (navigator.onLine) void flushOfflineQueue();
}
