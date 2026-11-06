import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";

export const GUEST_MAX_AGE = 30 * 24 * 60 * 60;
const mac = (text: string, secret: string) => createHmac("sha256", secret).update(`gha-r3-guest:${text}`).digest("hex");

export function newGuestToken(secret: string, now = Date.now()) {
  const text = `${randomBytes(32).toString("hex")}.${Math.floor(now / 1000) + GUEST_MAX_AGE}`;
  return `${text}.${mac(text, secret)}`;
}

export function guestOwner(token: string | undefined, secret: string, now = Date.now()): string | undefined {
  if (!token || !/^[a-f0-9]{64}\.\d{10}\.[a-f0-9]{64}$/.test(token)) return;
  const [nonce, expiry, signature] = token.split(".");
  const seconds = Math.floor(now / 1000);
  if (Number(expiry) <= seconds || Number(expiry) > seconds + GUEST_MAX_AGE) return;
  if (!timingSafeEqual(Buffer.from(signature, "hex"), Buffer.from(mac(`${nonce}.${expiry}`, secret), "hex"))) return;
  return createHash("sha256").update(nonce).digest("hex");
}

export function canAccessChat(conversation: { userId: string | null; metadata?: unknown }, userId?: string, owner?: string) {
  if (conversation.userId) return conversation.userId === userId;
  const metadata = conversation.metadata as { researchGuestOwner?: unknown } | null;
  if (metadata && Object.hasOwn(metadata, "researchGuestOwner")) {
    return typeof metadata.researchGuestOwner === "string" && metadata.researchGuestOwner === owner;
  }
  // Preserve legacy anonymous conversations; new R3 chats always have an owner.
  return true;
}
