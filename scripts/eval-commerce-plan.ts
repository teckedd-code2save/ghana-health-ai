import { buildCommerceActionPlan, type CommerceActionPlan } from "../src/lib/commerce-plan";
import type { CommerceUnderstanding } from "../src/lib/understand";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const cases: Array<{
  id: string;
  commerce: CommerceUnderstanding;
  status: CommerceActionPlan["status"];
  nextAction: CommerceActionPlan["nextAction"];
}> = [
  {
    id: "missing-item",
    commerce: {
      action: "buy",
      fulfillment: "unknown",
      confidence: 0.45,
      source: "deterministic",
    },
    status: "needs_clarification",
    nextAction: "ask_item",
  },
  {
    id: "buy-missing-quantity",
    commerce: {
      action: "buy",
      item: "tomatoes",
      fulfillment: "unknown",
      confidence: 0.81,
      source: "deterministic",
    },
    status: "needs_clarification",
    nextAction: "ask_quantity",
  },
  {
    id: "find-ready-search",
    commerce: {
      action: "find",
      item: "tomatoes",
      location: "Madina",
      fulfillment: "unknown",
      confidence: 0.89,
      source: "deterministic",
    },
    status: "ready_for_search",
    nextAction: "search_local_catalog",
  },
  {
    id: "delivery-ready-draft",
    commerce: {
      action: "order",
      item: "tomato",
      quantity: "2 kg",
      location: "Adenta",
      fulfillment: "delivery",
      confidence: 0.92,
      source: "deterministic",
    },
    status: "ready_for_order_draft",
    nextAction: "draft_order",
  },
];

for (const item of cases) {
  const plan = buildCommerceActionPlan(item.commerce);
  assertOk(plan, `${item.id}: missing plan`);
  assertOk(plan.status === item.status, `${item.id}: expected ${item.status}, got ${plan.status}`);
  assertOk(
    plan.nextAction === item.nextAction,
    `${item.id}: expected ${item.nextAction}, got ${plan.nextAction}`,
  );
  console.log(`ok ${item.id} status=${plan.status} next=${plan.nextAction}`);
}

export {};
