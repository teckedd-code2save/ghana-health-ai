"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { HeartPulse, Languages, Mic, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { useLang, type AppLang } from "@/components/lang-provider";
import { useAsrModel, type AsrModel } from "@/lib/asr-model-store";
import { SessionDrawer } from "@/components/session-drawer";

const menu = [
  { href: "/login", label: "Account", icon: User },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { lang, setLang, labels } = useLang();
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  // A/B: Whisper v6 (default, production) vs DONDO CTC (research endpoint)
  const [asrModel, changeAsrModel] = useAsrModel();

  useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", dismiss);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", dismiss);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div className="relative flex min-h-screen flex-col overflow-x-hidden">
      <div className="kente-bar" />

      <Link href="/" className="app-mark" aria-label="Ghana Health home">
        <HeartPulse className="h-4 w-4" />
      </Link>
      <span className="research-preview" title="Experimental language research; not medical care">
        Research Preview
      </span>

      <div ref={menuRef} className={cn("app-menu", open && "app-menu--open")}>
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

          <div className="app-menu__language">
            <Mic className="h-4 w-4 shrink-0" />
            <select
              value={asrModel}
              onChange={(e) => changeAsrModel(e.target.value as AsrModel)}
              aria-label="ASR model"
            >
              <option value="v6">ASR v6 (prod)</option>
              <option value="dondo">DONDO β (research)</option>
            </select>
          </div>

          <SessionDrawer onNavigate={() => setOpen(false)} />

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
