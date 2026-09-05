export const researchIntentOntology = [
  "health_symptom_report",
  "health_followup",
  "health_facility_search",
  "health_medication_question",
  "health_general_question",
  "health_emergency_report",
  "commerce_product_search",
  "commerce_purchase_request",
  "commerce_order_followup",
  "commerce_general_question",
  "general_statement",
  "general_question",
  "unclear_fragment",
] as const;

const researchIntentSet = new Set<string>(researchIntentOntology);

export function isResearchIntent(value: string) {
  return researchIntentSet.has(value);
}
