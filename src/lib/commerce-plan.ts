import type { CommerceUnderstanding } from "@/lib/understand";

export type CommerceActionPlan = {
  status: "needs_clarification" | "ready_for_search" | "ready_for_order_draft" | "unsupported";
  nextAction:
    | "ask_item"
    | "ask_quantity"
    | "ask_location"
    | "search_local_catalog"
    | "draft_order"
    | "explain_not_connected";
  missing: Array<"item" | "quantity" | "location">;
  searchQuery?: string;
  orderDraft?: {
    item: string;
    quantity?: string;
    location?: string;
    fulfillment: "delivery" | "pickup" | "unknown";
  };
  messageHint: string;
};

export function buildCommerceActionPlan(
  commerce: CommerceUnderstanding | undefined,
): CommerceActionPlan | undefined {
  if (!commerce) return undefined;

  if (!commerce.item) {
    return {
      status: "needs_clarification",
      nextAction: "ask_item",
      missing: ["item"],
      messageHint: "Ask which item the user wants.",
    };
  }

  const missing: CommerceActionPlan["missing"] = [];
  if (commerce.action === "buy" || commerce.action === "order") {
    if (!commerce.quantity) missing.push("quantity");
    if (commerce.fulfillment === "delivery" && !commerce.location) missing.push("location");
  }

  if (missing.length) {
    return {
      status: "needs_clarification",
      nextAction: missing[0] === "quantity" ? "ask_quantity" : "ask_location",
      missing,
      searchQuery: commerce.item,
      messageHint:
        missing[0] === "quantity"
          ? "Ask how much or how many the user wants."
          : "Ask where delivery or pickup should happen.",
    };
  }

  if (commerce.action === "price" || commerce.action === "availability" || commerce.action === "find") {
    return {
      status: "ready_for_search",
      nextAction: "search_local_catalog",
      missing,
      searchQuery: commerce.item,
      messageHint: "Search only connected/local catalog sources; do not invent prices or availability.",
    };
  }

  if (commerce.action === "buy" || commerce.action === "order") {
    return {
      status: "ready_for_order_draft",
      nextAction: "draft_order",
      missing,
      searchQuery: commerce.item,
      orderDraft: {
        item: commerce.item,
        quantity: commerce.quantity,
        location: commerce.location,
        fulfillment: commerce.fulfillment ?? "unknown",
      },
      messageHint: "Prepare an order draft, then confirm before any checkout/payment action.",
    };
  }

  return {
    status: "unsupported",
    nextAction: "explain_not_connected",
    missing,
    searchQuery: commerce.item,
    messageHint: "Explain that live marketplace ordering is not connected yet and ask one useful next question.",
  };
}
