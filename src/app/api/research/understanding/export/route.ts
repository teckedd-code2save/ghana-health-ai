import { getEnv } from "@/config/env";
import { jsonError, jsonOk } from "@/lib/api";
import { getSessionUser } from "@/lib/auth";
import { buildUnderstandingTrainingExport } from "@/lib/research-understanding-store";

export const dynamic = "force-dynamic";

async function canAccessResearchExport() {
  const env = getEnv();
  const user = await getSessionUser();
  const role = user?.role?.toLowerCase();
  return (
    env.NODE_ENV !== "production" ||
    env.RESEARCH_REVIEW_ENABLED ||
    role === "admin" ||
    role === "researcher"
  );
}

export async function GET() {
  if (!(await canAccessResearchExport())) {
    return jsonError("Research export access is not enabled for this account.", 403);
  }

  const payload = await buildUnderstandingTrainingExport();
  const bySplit = {
    train: payload.rows.filter((row) => row.split === "train"),
    dev: payload.rows.filter((row) => row.split === "dev"),
    test: payload.rows.filter((row) => row.split === "test"),
  };
  return jsonOk({
    ...payload,
    rows: bySplit,
    ready: payload.readiness.ready,
    message:
      payload.readiness.ready
        ? "Reviewed corpus rows are ready for training export."
        : "Reviewed corpus rows have not passed the training readiness gate yet.",
  });
}
