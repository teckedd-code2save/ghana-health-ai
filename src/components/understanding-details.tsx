import type { UnderstandResult } from "@/lib/understand";

export type UnderstandingDetailsData = {
  intent?: string;
  comprehension?: UnderstandResult["comprehension"];
  synthesis?: Partial<NonNullable<UnderstandResult["synthesis"]>>;
};

export function UnderstandingDetails({ data }: { data?: UnderstandingDetailsData | null }) {
  if (!data?.comprehension && !data?.synthesis) return null;
  const comprehension = data.comprehension;
  const prediction = comprehension?.model;
  const adapter = data.synthesis?.understandingModel;
  const mode = adapter?.mode === "assist_v1" ? "Research v1" : adapter?.mode === "assist" ? "Research v0" : adapter?.mode === "shadow" ? "Stable" : "Mode not recorded";
  const row = (label: string, value?: string | null) => value ? (
    <div><dt className="font-medium">{label}</dt><dd className="mt-1 whitespace-pre-wrap break-words">{value}</dd></div>
  ) : null;
  return (
    <details className="mt-2 max-w-xl rounded-lg border border-current/15 px-3 py-2 text-xs leading-relaxed">
      <summary className="cursor-pointer rounded focus-visible:outline-2 focus-visible:outline-offset-4">What the model understood</summary>
      <div className="mt-3 space-y-4">
        <p className="opacity-70">Structured interpretation, not private reasoning. This interpretation can be wrong.</p>
        <section aria-label="Research interpretation">
          <h4 className="font-semibold">{mode} · research interpretation</h4>
          <p className="mt-1 opacity-70">{adapter?.used ? "Adapter hint used for this reply." : adapter?.mode === "shadow" ? "No research adapter hint used for this reply." : "No adapter hint recorded for this reply."}</p>
          {prediction && <dl className="mt-2 space-y-2">
            {row("Normalized Twi", prediction.normalizedTwi)}
            {row("Meaning", prediction.naturalEnglish)}
            {row("Literal translation", prediction.literalEnglish)}
            {row("Intent", prediction.intent)}
            {row("Uncertainty", prediction.ambiguities)}
            {prediction.requiresClarification !== undefined && row("Needs clarification", prediction.requiresClarification ? "Yes" : "No")}
            {prediction.entities && Object.keys(prediction.entities).length > 0 && row("Details identified", JSON.stringify(prediction.entities, null, 2))}
            {row("Adapter model", prediction.model ?? adapter?.model)}
          </dl>}
        </section>
        <section aria-label="Response interpretation">
          <h4 className="font-semibold">Final response interpretation</h4>
          <dl className="mt-2 space-y-2">
            {row("Meaning", comprehension?.meaning)}
            {row("Intent", data.intent)}
            {comprehension && row("Reported understanding", comprehension.understood ? "Understood" : "Uncertain")}
            {row("Uncertainty", comprehension?.uncertaintyReason)}
            {row("Response model", data.synthesis?.model)}
            {data.synthesis?.mode === "degraded_fallback" && row("Response route", "Degraded fallback")}
          </dl>
        </section>
      </div>
    </details>
  );
}
