import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import { understandUtterance } from "../src/lib/understand";
import type { LanguageCode } from "@prisma/client";

type Fixture = {
  id: string;
  focus: "health" | "commerce";
  language: LanguageCode;
  text: string;
  memory?: unknown;
  transcript?: {
    language?: string;
    languageProbability?: number;
    duration?: number;
    rms?: number;
  };
  expectedIntent?: "HEALTH" | "ECOMMERCE" | "GENERAL" | "UNKNOWN";
  expectedSeverity?: "LOW" | "MEDIUM" | "HIGH" | "EMERGENCY";
  expectedEscalate?: boolean;
  expectedClarifying?: boolean;
  expectedHealth?: {
    planStatus?: "needs_clarification" | "self_care" | "clinic_recommended" | "urgent_referral";
    nextAction?:
      | "ask_clarifying_question"
      | "give_general_guidance"
      | "recommend_clinic"
      | "recommend_urgent_care";
    urgency?: "routine" | "soon" | "urgent" | "emergency";
  };
  expectedShoppingIntent?: string;
  expectedCommerce?: {
    action?: "buy" | "order" | "find" | "price" | "availability" | "unknown";
    itemIncludes?: string;
    quantityIncludes?: string;
    locationIncludes?: string;
    fulfillment?: "delivery" | "pickup" | "unknown";
    planStatus?: "needs_clarification" | "ready_for_search" | "ready_for_order_draft" | "unsupported";
    nextAction?:
      | "ask_item"
      | "ask_quantity"
      | "ask_location"
      | "search_local_catalog"
      | "draft_order"
      | "explain_not_connected";
  };
  forbiddenTerms?: string[];
};

const fixturePath =
  process.env.UNDERSTANDING_FIXTURES ||
  path.join(process.cwd(), "scripts", "understanding-fixtures.json");
const limit = Number(
  process.env.EVAL_LIMIT || process.argv.find((arg) => arg.startsWith("--limit="))?.split("=")[1] || 0,
);
const requireLlm = process.env.EVAL_REQUIRE_LLM === "1" || process.argv.includes("--require-llm");

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function looksClarifying(reply: string) {
  const lower = reply.toLowerCase();
  return (
    reply.includes("?") ||
    lower.includes("repeat") ||
    lower.includes("clarify") ||
    lower.includes("bio") ||
    lower.includes("san ka") ||
    lower.includes("mente") ||
    lower.includes("wopɛ sɛ")
  );
}

async function main() {
  const fixtures = JSON.parse(await fs.readFile(fixturePath, "utf8")) as Fixture[];
  const selected = limit > 0 ? fixtures.slice(0, limit) : fixtures;
  const results: { id: string; ok: boolean; error?: string }[] = [];

  for (const fixture of selected) {
    try {
      const result = await understandUtterance({
        text: fixture.text,
        language: fixture.language,
        focus: fixture.focus,
        memory: fixture.memory,
        transcript: {
          mode: "asr",
          language: fixture.transcript?.language,
          languageProbability: fixture.transcript?.languageProbability,
          duration: fixture.transcript?.duration,
          rms: fixture.transcript?.rms,
        },
      });

      if (requireLlm) {
        assertOk(result.engine === "llm", `${fixture.id}: expected model-first engine, got ${result.engine}`);
      } else {
        assertOk(
          result.engine === "llm" || result.engine === "fallback",
          `${fixture.id}: expected llm or fallback engine, got ${result.engine}`,
        );
      }
      assertOk(result.retrieve?.engine === "none", `${fixture.id}: retrieval should be disabled`);
      assertOk(result.reply.trim().length > 6, `${fixture.id}: empty reply`);

      if (fixture.expectedIntent) {
        assertOk(
          result.intent === fixture.expectedIntent,
          `${fixture.id}: expected intent ${fixture.expectedIntent}, got ${result.intent}`,
        );
      }
      if (fixture.expectedSeverity) {
        assertOk(
          result.severity === fixture.expectedSeverity,
          `${fixture.id}: expected severity ${fixture.expectedSeverity}, got ${result.severity}`,
        );
      }
      if (typeof fixture.expectedEscalate === "boolean") {
        assertOk(
          result.escalate === fixture.expectedEscalate,
          `${fixture.id}: expected escalate=${fixture.expectedEscalate}, got ${result.escalate}`,
        );
      }
      if (fixture.expectedClarifying) {
        assertOk(looksClarifying(result.reply), `${fixture.id}: expected clarification, got ${result.reply}`);
      }
      if (fixture.expectedHealth) {
        const health = result.health;
        assertOk(health, `${fixture.id}: expected health plan`);
        if (fixture.expectedHealth.planStatus) {
          assertOk(
            health.plan.status === fixture.expectedHealth.planStatus,
            `${fixture.id}: expected health plan ${fixture.expectedHealth.planStatus}, got ${health.plan.status}`,
          );
        }
        if (fixture.expectedHealth.nextAction) {
          assertOk(
            health.plan.nextAction === fixture.expectedHealth.nextAction,
            `${fixture.id}: expected health next action ${fixture.expectedHealth.nextAction}, got ${health.plan.nextAction}`,
          );
        }
        if (fixture.expectedHealth.urgency) {
          assertOk(
            health.plan.urgency === fixture.expectedHealth.urgency,
            `${fixture.id}: expected health urgency ${fixture.expectedHealth.urgency}, got ${health.plan.urgency}`,
          );
        }
      }
      if (fixture.expectedShoppingIntent) {
        assertOk(
          result.reply.toLowerCase().includes(fixture.expectedShoppingIntent.toLowerCase()) ||
            fixture.text.toLowerCase().includes(fixture.expectedShoppingIntent.toLowerCase()),
          `${fixture.id}: shopping item not preserved: ${fixture.expectedShoppingIntent}`,
        );
      }
      if (fixture.expectedCommerce) {
        const commerce = result.commerce;
        assertOk(commerce, `${fixture.id}: expected commerce slots`);
        if (fixture.expectedCommerce.action) {
          assertOk(
            commerce.action === fixture.expectedCommerce.action,
            `${fixture.id}: expected action ${fixture.expectedCommerce.action}, got ${commerce.action}`,
          );
        }
        if (fixture.expectedCommerce.itemIncludes) {
          assertOk(
            commerce.item?.toLowerCase().includes(fixture.expectedCommerce.itemIncludes.toLowerCase()),
            `${fixture.id}: expected item containing ${fixture.expectedCommerce.itemIncludes}, got ${commerce.item}`,
          );
        }
        if (fixture.expectedCommerce.quantityIncludes) {
          assertOk(
            commerce.quantity
              ?.toLowerCase()
              .includes(fixture.expectedCommerce.quantityIncludes.toLowerCase()),
            `${fixture.id}: expected quantity containing ${fixture.expectedCommerce.quantityIncludes}, got ${commerce.quantity}`,
          );
        }
        if (fixture.expectedCommerce.locationIncludes) {
          assertOk(
            commerce.location
              ?.toLowerCase()
              .includes(fixture.expectedCommerce.locationIncludes.toLowerCase()),
            `${fixture.id}: expected location containing ${fixture.expectedCommerce.locationIncludes}, got ${commerce.location}`,
          );
        }
        if (fixture.expectedCommerce.fulfillment) {
          assertOk(
            commerce.fulfillment === fixture.expectedCommerce.fulfillment,
            `${fixture.id}: expected fulfillment ${fixture.expectedCommerce.fulfillment}, got ${commerce.fulfillment}`,
          );
        }
        if (fixture.expectedCommerce.planStatus) {
          assertOk(
            commerce.plan?.status === fixture.expectedCommerce.planStatus,
            `${fixture.id}: expected plan ${fixture.expectedCommerce.planStatus}, got ${commerce.plan?.status}`,
          );
        }
        if (fixture.expectedCommerce.nextAction) {
          assertOk(
            commerce.plan?.nextAction === fixture.expectedCommerce.nextAction,
            `${fixture.id}: expected next action ${fixture.expectedCommerce.nextAction}, got ${commerce.plan?.nextAction}`,
          );
        }
      }
      for (const term of fixture.forbiddenTerms ?? []) {
        assertOk(
          !result.reply.toLowerCase().includes(term.toLowerCase()),
          `${fixture.id}: invented/forbidden term in reply: ${term}`,
        );
      }

      console.log(
        `ok ${fixture.id} focus=${fixture.focus} intent=${result.intent} severity=${result.severity} escalate=${result.escalate}`,
      );
      results.push({ id: fixture.id, ok: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`fail ${fixture.id}: ${message}`);
      results.push({ id: fixture.id, ok: false, error: message });
    }
  }

  if (results.some((result) => !result.ok)) process.exit(1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
