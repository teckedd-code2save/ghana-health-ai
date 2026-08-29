import { getEnv } from "@/config/env";
import { getSessionUser } from "@/lib/auth";
import { buildUnderstandingReviewSheetCsv } from "@/lib/research-understanding-store";

export const dynamic = "force-dynamic";

async function canAccessResearchReviewSheet() {
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
  if (!(await canAccessResearchReviewSheet())) {
    return Response.json({ error: "Research review access is not enabled for this account." }, { status: 403 });
  }

  const csv = await buildUnderstandingReviewSheetCsv();
  return new Response(csv, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": 'attachment; filename="understanding-review-sheet.v0.csv"',
      "Cache-Control": "no-store",
    },
  });
}
