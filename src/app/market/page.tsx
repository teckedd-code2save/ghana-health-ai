import { MarketPanel } from "@/components/market-panel";

export default function MarketPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="display text-2xl md:text-3xl">Market</h1>
        <p className="mt-1.5 text-sm text-[var(--fg-muted)]">
          Everyday staples and essentials — order when you’re ready.
        </p>
      </div>
      <MarketPanel />
    </div>
  );
}
