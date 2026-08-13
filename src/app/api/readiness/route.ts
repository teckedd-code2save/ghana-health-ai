import { jsonOk } from "@/lib/api";
import { getProductReadiness } from "@/lib/readiness";

export async function GET() {
  const readiness = await getProductReadiness();
  return jsonOk(readiness, {
    status: readiness.status === "blocked" ? 503 : 200,
  });
}
