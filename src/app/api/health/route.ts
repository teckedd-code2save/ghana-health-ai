import { jsonOk } from "@/lib/api";
import { prisma } from "@/db/prisma";
import { getProductReadiness } from "@/lib/readiness";

export async function GET() {
  let db = false;
  try {
    await prisma.$queryRaw`SELECT 1`;
    db = true;
  } catch {
    db = false;
  }

  const readiness = await getProductReadiness();

  return jsonOk({
    ok: db,
    service: "ghana-health-ai",
    version: "0.1.0",
    readiness: readiness.status,
    dependencies: {
      db,
    },
    ts: new Date().toISOString(),
  });
}
