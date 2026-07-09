import { prisma } from "@/db/prisma";
import { jsonOk } from "@/lib/api";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const q = searchParams.get("q")?.trim();
  const category = searchParams.get("category")?.trim();

  const products = await prisma.product.findMany({
    where: {
      isActive: true,
      ...(category ? { category } : {}),
      ...(q
        ? {
            OR: [
              { nameEn: { contains: q, mode: "insensitive" } },
              { nameTw: { contains: q, mode: "insensitive" } },
              { tags: { has: q.toLowerCase() } },
              { sku: { contains: q, mode: "insensitive" } },
            ],
          }
        : {}),
    },
    orderBy: { nameEn: "asc" },
  });

  return jsonOk({
    products: products.map((p) => ({
      ...p,
      priceGhs: Number(p.priceGhs),
    })),
  });
}
