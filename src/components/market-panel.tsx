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
    const t = setTimeout(() => {
      void load();
    }, 0);
    return () => clearTimeout(t);
  }, [load]);

  useEffect(() => {
    const t = setTimeout(() => {
      void fetch("/api/config")
        .then((r) => r.json())
        .then((d) => setPaystackEnabled(Boolean(d.paystack)))
        .catch(() => setPaystackEnabled(false));
    }, 0);
    return () => clearTimeout(t);
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
      if (!res.ok) throw new Error(data.error || "Couldn’t add");
      setCart(data.cart);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "Couldn’t add");
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
      if (!res.ok) throw new Error(data.error || "Checkout failed");
      if (data.payment?.authorizationUrl) {
        setMessage("Opening payment…");
        window.location.href = data.payment.authorizationUrl as string;
        return;
      }
      setMessage(`Order placed · ${formatGhs(data.order.totalGhs)}`);
      await load(q);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "Checkout failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fade-up grid gap-6 lg:grid-cols-[1.45fr_1fr]">
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
            placeholder="Search rice, soap, medicine…"
            className="field flex-1"
          />
          <button type="submit" className="btn btn-primary shrink-0">
            Search
          </button>
        </form>

        {loading ? (
          <div className="flex items-center gap-2 py-12 text-[var(--fg-muted)]">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading…
          </div>
        ) : products.length === 0 ? (
          <p className="py-10 text-center text-sm text-[var(--fg-muted)]">No items found.</p>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2">
            {products.map((p) => (
              <article key={p.id} className="surface rounded-[var(--radius)] p-4">
                <p className="text-[11px] uppercase tracking-wider text-[var(--fg-subtle)]">
                  {p.category}
                </p>
                <h3 className="display mt-1 text-lg">{p.nameTw}</h3>
                <p className="text-sm text-[var(--fg-muted)]">{p.nameEn}</p>
                <div className="mt-4 flex items-end justify-between gap-2">
                  <div>
                    <p className="text-lg font-semibold text-[var(--accent)]">
                      {formatGhs(p.priceGhs)}
                    </p>
                    <p className="text-xs text-[var(--fg-subtle)]">
                      {p.stock > 0 ? "In stock" : "Out of stock"}
                    </p>
                  </div>
                  <button
                    disabled={busy || p.stock < 1}
                    onClick={() => void addToCart(p.id)}
                    className="btn btn-teal px-4 py-2 text-sm"
                  >
                    Add
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <aside className="surface h-fit rounded-[var(--radius)] p-5">
        <div className="mb-4 flex items-center gap-2">
          <ShoppingCart className="h-4 w-4 text-[var(--accent)]" />
          <h2 className="font-medium">Cart</h2>
        </div>
        {!cart || cart.items.length === 0 ? (
          <p className="text-sm text-[var(--fg-muted)]">Your cart is empty.</p>
        ) : (
          <ul className="space-y-3">
            {cart.items.map((item) => (
              <li key={item.id} className="flex items-center justify-between gap-2 text-sm">
                <div className="min-w-0">
                  <p className="truncate">{item.product.nameTw}</p>
                  <p className="text-xs text-[var(--fg-subtle)]">{formatGhs(item.lineTotal)}</p>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    className="icon-action h-8 w-8"
                    onClick={() => void updateQty(item.id, item.quantity - 1)}
                    disabled={busy}
                    aria-label="Less"
                  >
                    <Minus className="h-3.5 w-3.5" />
                  </button>
                  <span className="w-6 text-center tabular-nums">{item.quantity}</span>
                  <button
                    className="icon-action h-8 w-8"
                    onClick={() => void updateQty(item.id, item.quantity + 1)}
                    disabled={busy}
                    aria-label="More"
                  >
                    <Plus className="h-3.5 w-3.5" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
        <div className="mt-5 border-t border-white/[0.06] pt-4">
          <div className="mb-4 flex justify-between text-sm">
            <span className="text-[var(--fg-muted)]">Total</span>
            <span className="font-semibold tabular-nums">{formatGhs(cart?.totalGhs ?? 0)}</span>
          </div>
          <input
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            placeholder="Mobile money number"
            className="field mb-2 rounded-[var(--radius-sm)]"
          />
          <select
            value={payMethod}
            onChange={(e) => setPayMethod(e.target.value as "MOMO" | "MOCK" | "CASH")}
            className="field mb-3 rounded-[var(--radius-sm)]"
          >
            <option value="MOMO">Mobile Money</option>
            <option value="CASH">Pay on pickup</option>
            {!paystackEnabled && <option value="MOCK">Pay later (demo)</option>}
          </select>
          <button
            disabled={busy || !cart?.items.length}
            onClick={() => void checkout()}
            className="btn btn-primary w-full"
          >
            {busy ? "Working…" : "Checkout"}
          </button>
          {message && <p className="mt-3 text-sm text-[var(--fg-muted)]">{message}</p>}
        </div>
      </aside>
    </div>
  );
}
