import { getEnv } from "@/config/env";
import { getSessionUser } from "@/lib/auth";
import {
  buildUnderstandingReviewSheetCsv,
  buildUnderstandingTrainingExport,
  parseUnderstandingReviewSheetCsv,
  saveUnderstandingReview,
} from "@/lib/research-understanding-store";

export const dynamic = "force-dynamic";

async function getResearchReviewer() {
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

export async function GET(request: Request) {
  if (!(await getResearchReviewer())) {
    return Response.json({ error: "Research review access is not enabled for this account." }, { status: 403 });
  }

  const scope = new URL(request.url).searchParams.get("scope");
  const reviewScope = scope === "minimum-training" ? "minimum-training" : "all";
  const csv = await buildUnderstandingReviewSheetCsv(reviewScope);
  return new Response(csv, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition":
        reviewScope === "minimum-training"
          ? 'attachment; filename="understanding-minimum-training-review.v0.csv"'
          : 'attachment; filename="understanding-review-sheet.v0.csv"',
      "Cache-Control": "no-store",
    },
  });
}

export async function POST(request: Request) {
  const reviewer = await getResearchReviewer();
  if (!reviewer) {
    return Response.json({ error: "Research review access is not enabled for this account." }, { status: 403 });
  }

  const contentType = request.headers.get("content-type") ?? "";
  let csv = "";
  if (contentType.includes("multipart/form-data")) {
    const form = await request.formData();
    const file = form.get("file");
    const text = form.get("csv");
    if (file instanceof File) {
      csv = await file.text();
    } else if (typeof text === "string") {
      csv = text;
    }
  } else {
    csv = await request.text();
  }

  if (!csv.trim()) {
    return Response.json({ error: "Upload a non-empty review CSV." }, { status: 400 });
  }

  try {
    const imported = parseUnderstandingReviewSheetCsv(csv, reviewer);
    const saved = [];
    for (const review of imported.reviews) {
      saved.push(await saveUnderstandingReview(review, reviewer));
    }
    const exportPayload = await buildUnderstandingTrainingExport();
    return Response.json({
      imported: imported.reviews.length,
      skipped: imported.skipped,
      saved: saved.length,
      accepted: exportPayload.accepted,
      splits: exportPayload.splits,
      readiness: exportPayload.readiness,
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : "Review sheet could not be imported." },
      { status: 400 },
    );
  }
}
