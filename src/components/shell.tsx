"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { HeartPulse, MessageCircle, ShoppingBag, Mic, Home, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { useLang, type AppLang } from "@/components/lang-provider";

const nav = [
  { href: "/", label: "Home", icon: Home },
  { href: "/voice", label: "Speak", icon: Mic },
  { href: "/chat", label: "Chat", icon: MessageCircle },
  { href: "/market", label: "Market", icon: ShoppingBag },
  { href: "/login", label: "Account", icon: User },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { lang, setLang, labels } = useLang();

  return (
    <div className="flex min-h-screen flex-col">
      <div className="kente-bar" />
      <header className="sticky top-0 z-40 border-b border-white/[0.06] bg-[rgba(7,21,16,0.82)] backdrop-blur-xl">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-3 px-4 py-3.5">
          <Link href="/" className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-2xl bg-[var(--teal)] text-[#062419] shadow-lg shadow-emerald-950/40">
              <HeartPulse className="h-4.5 w-4.5" />
            </span>
            <div className="leading-tight">
              <p className="display text-[1.05rem]">Ghana Health</p>
            </div>
          </Link>

          <div className="flex items-center gap-2">
            <label className="sr-only" htmlFor="lang">
              Language
            </label>
            <select
              id="lang"
              value={lang}
              onChange={(e) => setLang(e.target.value as AppLang)}
              className="rounded-full border border-white/10 bg-black/25 px-2.5 py-1.5 text-xs text-[var(--fg-muted)] outline-none focus:ring-1 focus:ring-[var(--accent)]"
            >
              {(Object.keys(labels) as AppLang[]).map((code) => (
                <option key={code} value={code}>
                  {labels[code]}
                </option>
              ))}
            </select>

            <nav className="hidden items-center gap-0.5 md:flex">
              {nav.map((item) => {
                const active = pathname === item.href;
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-full px-3 py-2 text-sm transition",
                      active
                        ? "bg-[var(--accent)] font-medium text-[#1a1400]"
                        : "text-[var(--fg-muted)] hover:bg-white/[0.04] hover:text-white",
                    )}
                  >
                    <Icon className="h-3.5 w-3.5" />
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-6 pb-28 md:py-10 md:pb-12">
        {children}
      </main>

      <nav className="fixed inset-x-0 bottom-0 z-40 border-t border-white/[0.06] bg-[rgba(7,21,16,0.92)] backdrop-blur-xl md:hidden pb-[var(--safe-bottom)]">
        <div className="mx-auto grid max-w-lg grid-cols-5 gap-0.5 px-2 py-2">
          {nav.map((item) => {
            const active = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex flex-col items-center gap-1 rounded-xl px-1 py-2 text-[10px] font-medium",
                  active ? "text-[var(--accent)]" : "text-[var(--fg-subtle)]",
                )}
              >
                <Icon className="h-5 w-5" strokeWidth={active ? 2.25 : 1.75} />
                {item.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
}
