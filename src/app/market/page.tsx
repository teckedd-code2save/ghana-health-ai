import { MarketPanel } from "@/components/market-panel";

export default function MarketPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-3xl">Market</h1>
        <p className="text-sm text-[var(--fg-muted)]">
          Staples, OTC meds, household — prices from Postgres, cart and mock MoMo checkout.
        </p>
      </div>
      <MarketPanel />
    </div>
  );
}
