"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { HeartPulse, History, Languages, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { useLang, type AppLang } from "@/components/lang-provider";

const menu = [
  { href: "/chat", label: "Sessions", icon: History },
  { href: "/login", label: "Account", icon: User },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { lang, setLang, labels } = useLang();
  const [open, setOpen] = useState(false);

  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden">
      <div className="kente-bar" />

      <Link href="/" className="app-mark" aria-label="Ghana Health home">
        <HeartPulse className="h-4 w-4" />
      </Link>

      <div className={cn("app-menu", open && "app-menu--open")}>
        <button
          type="button"
          className="app-menu__button"
          aria-label="Open settings and sessions"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          <span className="app-menu__glyph" aria-hidden>
            <span />
            <span />
            <span />
          </span>
        </button>

        <div className="app-menu__panel">
          <div className="app-menu__language">
            <Languages className="h-4 w-4 shrink-0" />
            <select
              value={lang}
              onChange={(e) => setLang(e.target.value as AppLang)}
              aria-label="Language"
            >
              {(Object.keys(labels) as AppLang[]).map((code) => (
                <option key={code} value={code}>
                  {labels[code]}
                </option>
              ))}
            </select>
          </div>

          {menu.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "app-menu__item",
                  pathname === item.href && "app-menu__item--active",
                )}
                onClick={() => setOpen(false)}
              >
                <Icon className="h-4 w-4 shrink-0" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>
      </div>

      <main className="relative z-10 flex min-h-[calc(100vh-3px)] w-full flex-1 px-4 py-5 pb-[calc(1.25rem+var(--safe-bottom))] md:px-8 md:py-8">
        {children}
      </main>
    </div>
  );
}
