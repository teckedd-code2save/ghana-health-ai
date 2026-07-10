import { prisma } from "@/db/prisma";
import { getSessionUser } from "@/lib/auth";
import { jsonError, jsonOk } from "@/lib/api";

export async function GET(req: Request) {
  const user = await getSessionUser();
  if (!user || user.role !== "admin") {
    return jsonError("Admin only", 403);
  }
  const { searchParams } = new URL(req.url);
  const take = Math.min(Number(searchParams.get("limit") || 50), 200);
  const logs = await prisma.auditLog.findMany({
    orderBy: { createdAt: "desc" },
    take,
  });
  return jsonOk({ logs });
}
