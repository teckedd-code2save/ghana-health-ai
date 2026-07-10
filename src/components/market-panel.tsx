"use client";

import { useCallback, useEffect, useState } from "react";
import { Loader2, Minus, Plus, ShoppingCart } from "lucide-react";
import { formatGhs } from "@/lib/utils";

type Product = {
  id: string;
  sku: string;
  nameEn: string;
  nameTw: string;
  category: string;
  priceGhs: number;
  unit: string;
  stock: number;
};

type Cart = {
  id: string;
  items: {
    id: string;
    productId: string;
    quantity: number;
    lineTotal: number;
    product: { nameEn: string; nameTw: string; priceGhs: number };
  }[];
  totalGhs: number;
};

export function MarketPanel() {
  const [products, setProducts] = useState<Product[]>([]);
  const [cart, setCart] = useState<Cart | null>(null);
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [payMethod, setPayMethod] = useState<"MOMO" | "MOCK" | "CASH">("MOMO");
  const [phone, setPhone] = useState("");
  const [paystackEnabled, setPaystackEnabled] = useState(false);

  const load = useCallback(async (query = "") => {
    setLoading(true);
    try {
      const [pRes, cRes] = await Promise.all([
        fetch(`/api/products${query ? `?q=${encodeURIComponent(query)}` : ""}`),
        fetch("/api/cart"),
      ]);
      const pData = await pRes.json();
      const cData = await cRes.json();
      setProducts(pData.products ?? []);
      setCart(cData.cart ?? null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    void fetch("/api/config")
      .then((r) => r.json())
      .then((d) => setPaystackEnabled(Boolean(d.paystack)))
      .catch(() => setPaystackEnabled(false));
  }, []);

  async function addToCart(productId: string) {
    setBusy(true);
    setMessage(null);
    try {
      const res = await fetch("/api/cart", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ productId, quantity: 1 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setCart(data.cart);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "Failed");
    } finally {
      setBusy(false);
    }
  }

  async function updateQty(itemId: string, quantity: number) {
    setBusy(true);
    try {
      const res = await fetch("/api/cart", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ itemId, quantity }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setCart(data.cart);
    } finally {
      setBusy(false);
    }
  }

  async function checkout() {
    setBusy(true);
    setMessage(null);
    try {
      const method = paystackEnabled ? payMethod : payMethod === "MOMO" ? "MOCK" : payMethod;
      const res = await fetch("/api/orders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paymentMethod: method, phone: phone || undefined }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      if (data.payment?.authorizationUrl) {
        setMessage("Redirecting to Paystack MoMo / card…");
        window.location.href = data.payment.authorizationUrl as string;
        return;
      }
      setMessage(
        `Order ${data.order.status} · ${data.order.id.slice(0, 8)} · ${formatGhs(data.order.totalGhs)}`,
      );
      await load(q);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "Checkout failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <section className="space-y-4">
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void load(q);
          }}
        >
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search rice, paracetamol, soap…"
            className="flex-1 rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-sm outline-none focus:ring-1 focus:ring-[var(--accent)]"
          />
          <button
            type="submit"
            className="rounded-2xl bg-[var(--accent)] px-4 py-3 text-sm font-medium text-[#1a1400]"
          >
            Search
          </button>
        </form>

        {loading ? (
          <div className="flex items-center gap-2 text-[var(--fg-muted)]">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading market…
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2">
            {products.map((p) => (
              <article key={p.id} className="glass rounded-[var(--radius)] p-4">
                <p className="text-[10px] uppercase tracking-wider text-[var(--fg-muted)]">
                  {p.category} · {p.sku}
                </p>
                <h3 className="mt-1 font-[family-name:var(--font-display)] text-lg">{p.nameEn}</h3>
                <p className="text-sm text-[var(--fg-muted)]">{p.nameTw}</p>
                <div className="mt-3 flex items-center justify-between">
                  <div>
                    <p className="text-lg font-semibold text-[var(--accent)]">{formatGhs(p.priceGhs)}</p>
                    <p className="text-xs text-[var(--fg-muted)]">
                      {p.stock} {p.unit} in stock
                    </p>
                  </div>
                  <button
                    disabled={busy || p.stock < 1}
                    onClick={() => void addToCart(p.id)}
                    className="rounded-full bg-[var(--teal)] px-3 py-2 text-sm font-medium text-[#062419] disabled:opacity-40"
                  >
                    Add
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <aside className="glass h-fit rounded-[var(--radius)] p-4">
        <div className="mb-3 flex items-center gap-2">
          <ShoppingCart className="h-4 w-4 text-[var(--accent)]" />
          <h2 className="font-medium">Cart</h2>
        </div>
        {!cart || cart.items.length === 0 ? (
          <p className="text-sm text-[var(--fg-muted)]">Empty — add staples or OTC items.</p>
        ) : (
          <ul className="space-y-3">
            {cart.items.map((item) => (
              <li key={item.id} className="flex items-center justify-between gap-2 text-sm">
                <div>
                  <p>{item.product.nameEn}</p>
                  <p className="text-xs text-[var(--fg-muted)]">{formatGhs(item.lineTotal)}</p>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    className="rounded-full bg-white/5 p-1"
                    onClick={() => void updateQty(item.id, item.quantity - 1)}
                    disabled={busy}
                  >
                    <Minus className="h-3.5 w-3.5" />
                  </button>
                  <span className="w-6 text-center">{item.quantity}</span>
                  <button
                    className="rounded-full bg-white/5 p-1"
                    onClick={() => void updateQty(item.id, item.quantity + 1)}
                    disabled={busy}
                  >
                    <Plus className="h-3.5 w-3.5" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
        <div className="mt-4 border-t border-white/5 pt-4">
          <div className="mb-3 flex justify-between text-sm">
            <span className="text-[var(--fg-muted)]">Total</span>
            <span className="font-semibold">{formatGhs(cart?.totalGhs ?? 0)}</span>
          </div>
          <input
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            placeholder="MoMo phone e.g. 024…"
            className="mb-2 w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-[var(--accent)]"
          />
          <select
            value={payMethod}
            onChange={(e) => setPayMethod(e.target.value as "MOMO" | "MOCK" | "CASH")}
            className="mb-3 w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm"
          >
            <option value="MOMO">
              {paystackEnabled ? "Mobile Money (Paystack)" : "Mobile Money (mock — add Paystack keys)"}
            </option>
            <option value="MOCK">Demo pay (instant)</option>
            <option value="CASH">Cash on pickup</option>
          </select>
          <button
            disabled={busy || !cart?.items.length}
            onClick={() => void checkout()}
            className="w-full rounded-2xl bg-[var(--accent)] py-3 text-sm font-semibold text-[#1a1400] disabled:opacity-40"
          >
            {busy ? "Working…" : "Checkout"}
          </button>
          {message && <p className="mt-3 text-xs text-[var(--accent-soft)]">{message}</p>}
        </div>
      </aside>
    </div>
  );
}
