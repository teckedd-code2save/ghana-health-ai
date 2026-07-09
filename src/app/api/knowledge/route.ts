import { prisma } from "@/db/prisma";
import { jsonOk } from "@/lib/api";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const category = searchParams.get("category") ?? undefined;
  const articles = await prisma.knowledgeArticle.findMany({
    where: { isActive: true, ...(category ? { category } : {}) },
    orderBy: { titleEn: "asc" },
  });
  return jsonOk({ articles });
}
