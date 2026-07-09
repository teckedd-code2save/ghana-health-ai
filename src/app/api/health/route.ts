import { jsonOk } from "@/lib/api";

export async function GET() {
  return jsonOk({
    ok: true,
    service: "ghana-health-ai",
    version: "0.1.0",
    ts: new Date().toISOString(),
  });
}
