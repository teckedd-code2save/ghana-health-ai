import type { IntentType, LanguageCode, MemoryScope, Prisma } from "@prisma/client";
import { prisma } from "@/db/prisma";

export type MemoryOwner = {
  userId?: string | null;
  sessionId?: string | null;
};

export type AgentMemoryItem = {
  scope: MemoryScope;
  key: string;
  value: unknown;
  confidence: number;
  source: string;
};

export async function getAgentMemory(owner: MemoryOwner, scope?: MemoryScope) {
  if (!owner.userId && !owner.sessionId) return [];
  return prisma.agentMemory.findMany({
    where: {
      ...(scope ? { scope } : {}),
      OR: [
        ...(owner.userId ? [{ userId: owner.userId }] : []),
        ...(owner.sessionId ? [{ sessionId: owner.sessionId }] : []),
      ],
    },
    orderBy: [{ confidence: "desc" }, { lastObservedAt: "desc" }],
    take: 12,
  });
}

export function memoryPrompt(items: Awaited<ReturnType<typeof getAgentMemory>>) {
  if (!items.length) return null;
  return items.map((item) => ({
    scope: item.scope,
    key: item.key,
    value: item.value,
    confidence: item.confidence,
  }));
}

export async function rememberFromTurn(input: {
  owner: MemoryOwner;
  text: string;
  reply: string;
  language: LanguageCode;
  focus?: "health" | "commerce";
  intent: IntentType;
  severity?: string;
}) {
  const candidates = extractMemoryCandidates(input);
  if (!candidates.length) return [];

  const saved = [];
  for (const candidate of candidates) {
    saved.push(await upsertMemory(input.owner, candidate));
  }
  return saved;
}

async function upsertMemory(owner: MemoryOwner, item: AgentMemoryItem) {
  const where = {
    scope: item.scope,
    key: item.key,
    ...(owner.userId ? { userId: owner.userId } : { userId: null }),
    ...(owner.sessionId ? { sessionId: owner.sessionId } : { sessionId: null }),
  };
  const existing = await prisma.agentMemory.findFirst({ where });
  const data = {
    userId: owner.userId ?? null,
    sessionId: owner.sessionId ?? null,
    scope: item.scope,
    key: item.key,
    value: item.value as Prisma.InputJsonValue,
    confidence: item.confidence,
    source: item.source,
    lastObservedAt: new Date(),
  };
  if (existing) {
    return prisma.agentMemory.update({ where: { id: existing.id }, data });
  }
  return prisma.agentMemory.create({ data });
}

function extractMemoryCandidates(input: {
  text: string;
  reply: string;
  language: LanguageCode;
  focus?: "health" | "commerce";
  intent: IntentType;
  severity?: string;
}): AgentMemoryItem[] {
  const text = input.text.trim();
  const lower = text.toLowerCase();
  const items: AgentMemoryItem[] = [];

  if (input.language) {
    items.push({
      scope: "PROFILE",
      key: "last_spoken_language",
      value: input.language,
      confidence: 0.85,
      source: "asr",
    });
  }

  if (input.focus === "commerce" || input.intent === "ECOMMERCE") {
    const shoppingItem = extractShoppingItem(lower);
    if (shoppingItem) {
      items.push({
        scope: "COMMERCE",
        key: "last_requested_item",
        value: { item: shoppingItem, utterance: text },
        confidence: 0.72,
        source: "conversation",
      });
    }
    const location = extractLocation(lower);
    if (location) {
      items.push({
        scope: "COMMERCE",
        key: "preferred_shopping_location",
        value: { location, utterance: text },
        confidence: 0.68,
        source: "conversation",
      });
    }
  }

  if (input.focus === "health" || input.intent === "HEALTH") {
    items.push({
      scope: "HEALTH",
      key: "last_health_utterance",
      value: {
        utterance: text,
        severity: input.severity ?? "LOW",
      },
      confidence: 0.62,
      source: "conversation",
    });
  }

  return items;
}

function extractShoppingItem(lower: string) {
  const patterns = [
    /\b(?:buy|order|get|find)\s+([a-zɛɔŋ][\wɛɔŋ -]{1,40})/i,
    /\b(?:metɔ|mɛtɔ|me tɔ|mepɛ|me pɛ|mɛpɛ)\s+([a-zɛɔŋ][\wɛɔŋ -]{1,40})/i,
  ];
  for (const pattern of patterns) {
    const match = lower.match(pattern);
    const item = match?.[1]?.replace(/\b(no|bi|please|pls)\b/g, "").trim();
    if (item && item.length > 1) return item.slice(0, 48);
  }
  return null;
}

function extractLocation(lower: string) {
  const match = lower.match(/\b(?:at|in|near|around|wɔ|wo)\s+([a-zɛɔŋ][\wɛɔŋ -]{2,40})/i);
  const location = match?.[1]?.trim();
  return location ? location.slice(0, 48) : null;
}
