import { z } from "zod";
import { getEnv } from "@/config/env";
import { jsonError, jsonOk } from "@/lib/api";
import { getSessionUser } from "@/lib/auth";
import { prisma } from "@/db/prisma";
import { readCorpusCandidates } from "@/lib/research-understanding-store";

export const dynamic = "force-dynamic";

const MAX_AUDIO_BYTES = 10 * 1024 * 1024;
const metadataSchema = z.object({
  rowId: z.string().min(1).max(200),
  speakerId: z.string().trim().min(2).max(80).regex(/^[a-zA-Z0-9_-]+$/),
  transcript: z.string().trim().min(1).max(1500),
  language: z.enum(["tw", "en", "ga"]).default("tw"),
  durationMs: z.coerce.number().int().min(0).max(120_000).optional(),
  consent: z.literal("true"),
});

async function getReviewer() {
  const env = getEnv();
  const user = await getSessionUser();
  const role = user?.role?.toLowerCase();
  const allowed =
    env.NODE_ENV !== "production" ||
    env.RESEARCH_REVIEW_ENABLED ||
    role === "admin" ||
    role === "researcher";

  if (!allowed) return null;
  return user?.email ?? user?.phone ?? user?.id ?? "local_reviewer";
}

function isAudio(blob: Blob) {
  const name = blob instanceof File ? blob.name : "";
  const type = blob.type.toLowerCase();
  return type.startsWith("audio/") || type.includes("webm") || /\.(webm|wav|m4a|mp3|ogg)$/i.test(name);
}

export async function GET(request: Request) {
  const reviewer = await getReviewer();
  if (!reviewer) return jsonError("Research review access is not enabled for this account.", 403);
  const rowId = new URL(request.url).searchParams.get("rowId")?.trim();
  if (!rowId) return jsonError("rowId is required.", 400);

  const recordings = await prisma.researchCorpusRecording.findMany({
    where: { rowId },
    select: { id: true, speakerId: true, durationMs: true, createdAt: true },
    orderBy: { createdAt: "desc" },
  });
  return jsonOk({ recordings });
}

export async function POST(request: Request) {
  const reviewer = await getReviewer();
  if (!reviewer) return jsonError("Research review access is not enabled for this account.", 403);

  try {
    const form = await request.formData();
    const audio = form.get("audio");
    const parsed = metadataSchema.safeParse({
      rowId: form.get("rowId"),
      speakerId: form.get("speakerId"),
      transcript: form.get("transcript"),
      language: form.get("language") ?? "tw",
      durationMs: form.get("durationMs") ?? undefined,
      consent: form.get("consent"),
    });
    if (!parsed.success) {
      return jsonError(parsed.error.issues[0]?.message ?? "Invalid recording details.", 400);
    }
    if (!(audio instanceof Blob) || !isAudio(audio) || audio.size <= 0 || audio.size > MAX_AUDIO_BYTES) {
      return jsonError("Upload a voice clip under 10 MB.", 400);
    }

    const candidates = await readCorpusCandidates();
    if (!candidates.some((row) => row.id === parsed.data.rowId)) {
      return jsonError("Unknown research row.", 404);
    }

    const recording = await prisma.researchCorpusRecording.create({
      data: {
        rowId: parsed.data.rowId,
        speakerId: parsed.data.speakerId,
        transcript: parsed.data.transcript,
        language: parsed.data.language,
        mimeType: audio.type || "audio/webm",
        audioBytes: Buffer.from(await audio.arrayBuffer()),
        sizeBytes: audio.size,
        durationMs: parsed.data.durationMs,
        reviewer,
      },
      select: { id: true, speakerId: true, durationMs: true, createdAt: true },
    });

    return jsonOk({ recording });
  } catch (error) {
    console.error("[research-understanding-recordings]", error);
    return jsonError("Recording could not be saved.", 500);
  }
}
