import type { Prisma } from "@prisma/client";
import { prisma } from "@/db/prisma";

export async function writeAudit(input: {
  action: string;
  actorId?: string | null;
  entityType?: string | null;
  entityId?: string | null;
  ip?: string | null;
  meta?: Prisma.InputJsonValue;
}) {
  try {
    await prisma.auditLog.create({
      data: {
        action: input.action,
        actorId: input.actorId ?? null,
        entityType: input.entityType ?? null,
        entityId: input.entityId ?? null,
        ip: input.ip ?? null,
        meta: input.meta ?? undefined,
      },
    });
  } catch (e) {
    // Never fail the request on audit write
    console.error("[audit]", e);
  }
}
