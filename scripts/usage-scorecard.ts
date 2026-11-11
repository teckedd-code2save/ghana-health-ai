import { prisma } from "../src/db/prisma";

function argDays() {
  const raw = process.argv.find((arg) => arg.startsWith("--days="))?.split("=")[1]
    ?? process.argv[process.argv.indexOf("--days") + 1];
  const days = Number(raw || 30);
  return Number.isFinite(days) && days > 0 ? Math.floor(days) : 30;
}

function percentile(values: number[], p: number) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
}

function ratio(a: number, b: number) {
  return b ? Number((a / b).toFixed(4)) : null;
}

async function main() {
  const days = argDays();
  const since = new Date(Date.now() - days * 86400000);

  const [conversations, messages, feedback, reviews] = await Promise.all([
    prisma.conversation.findMany({
      where: { createdAt: { gte: since } },
      select: { userId: true, channel: true, language: true },
    }),
    prisma.message.findMany({
      where: { createdAt: { gte: since } },
      select: { role: true, language: true, latencyMs: true },
    }),
    prisma.asrFeedback.findMany({
      where: { createdAt: { gte: since } },
      select: { correctedTranscript: true, audioConsent: true, language: true },
    }),
    prisma.researchUnderstandingReview.findMany({
      where: { updatedAt: { gte: since } },
      select: { decision: true, reviewer: true },
    }),
  ]);

  const activeUsers = new Set(conversations.map((conversation) => conversation.userId).filter(Boolean));
  const userMessages = messages.filter((message) => message.role === "USER");
  const twiUserTurns = userMessages.filter((message) => message.language === "tw").length;
  const latencies = messages.map((message) => message.latencyMs).filter((value): value is number => value != null);
  const corrections = feedback.filter((item) => Boolean(item.correctedTranscript?.trim()));
  const reviewCounts = Object.fromEntries(
    [...new Set(reviews.map((review) => review.decision))].sort().map((decision) => [
      decision,
      reviews.filter((review) => review.decision === decision).length,
    ]),
  );

  const scorecard = {
    generatedAt: new Date().toISOString(),
    window: { days, since: since.toISOString() },
    productUsage: {
      conversations: conversations.length,
      authenticatedUsers: activeUsers.size,
      voiceChannelConversations: conversations.filter((conversation) => conversation.channel === "VOICE").length,
      totalUserTurns: userMessages.length,
      twiUserTurns,
      twiTurnShare: ratio(twiUserTurns, userMessages.length),
      responseLatencyMs: {
        median: percentile(latencies, 0.5),
        p95: percentile(latencies, 0.95),
      },
    },
    feedbackLoop: {
      asrFeedbackRows: feedback.length,
      correctedTranscripts: corrections.length,
      consentedAudioFeedback: feedback.filter((item) => item.audioConsent).length,
      correctionRate: ratio(corrections.length, feedback.length),
    },
    researchData: {
      reviewsUpdated: reviews.length,
      byDecision: reviewCounts,
    },
    definitions: {
      twiUserTurn: "USER message explicitly stored with language=tw",
      correction: "ASR feedback row containing a non-empty corrected transcript",
      consentedAudioFeedback: "ASR feedback row with audioConsent=true",
      researchReview: "ResearchUnderstandingReview updated inside the window; corpus-recording metrics live on the active response-research branch and are intentionally not mixed into the production-main scorecard",
    },
  };

  console.log(JSON.stringify(scorecard, null, 2));
  await prisma.$disconnect();
}

main().catch(async (error) => {
  console.error(error);
  await prisma.$disconnect();
  process.exit(1);
});
