"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { HeartPulse, MessageCircle, ShoppingBag, Mic, Home, LogIn } from "lucide-react";
import { cn } from "@/lib/utils";
import { useLang, type AppLang } from "@/components/lang-provider";

const nav = [
  { href: "/", label: "Home", icon: Home },
  { href: "/chat", label: "Chat", icon: MessageCircle },
  { href: "/voice", label: "Voice", icon: Mic },
  { href: "/market", label: "Market", icon: ShoppingBag },
  { href: "/login", label: "Account", icon: LogIn },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { lang, setLang, labels } = useLang();

  return (
    <div className="min-h-screen flex flex-col">
      <div className="kente-bar" />
      <header className="sticky top-0 z-40 border-b border-white/5 bg-[#0c1a14]/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
          <Link href="/" className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[var(--teal)] text-[#062419] shadow-lg shadow-emerald-900/40">
              <HeartPulse className="h-5 w-5" />
            </span>
            <div>
              <p className="font-[family-name:var(--font-display)] text-lg leading-none tracking-tight">
                Ghana Health AI
              </p>
              <p className="text-xs text-[var(--fg-muted)]">Voice-first · {labels[lang]}</p>
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
              className="rounded-full border border-white/10 bg-black/30 px-2.5 py-1.5 text-xs text-[var(--fg-muted)] outline-none focus:ring-1 focus:ring-[var(--accent)]"
            >
              {(Object.keys(labels) as AppLang[]).map((code) => (
                <option key={code} value={code}>
                  {labels[code]}
                </option>
              ))}
            </select>
            <nav className="hidden items-center gap-1 md:flex">
              {nav.map((item) => {
                const active = pathname === item.href;
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "inline-flex items-center gap-2 rounded-full px-3.5 py-2 text-sm transition",
                      active
                        ? "bg-[var(--accent)] text-[#1a1400] font-medium"
                        : "text-[var(--fg-muted)] hover:bg-white/5 hover:text-white",
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 pb-24 md:pb-8">{children}</main>

      <nav className="fixed inset-x-0 bottom-0 z-40 border-t border-white/5 bg-[#0c1a14]/95 backdrop-blur-md md:hidden">
        <div className="mx-auto grid max-w-lg grid-cols-5 gap-1 px-2 py-2">
          {nav.map((item) => {
            const active = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex flex-col items-center gap-1 rounded-xl px-1 py-2 text-[10px]",
                  active ? "text-[var(--accent)]" : "text-[var(--fg-muted)]",
                )}
              >
                <Icon className="h-5 w-5" />
                {item.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
}
