"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useSyncExternalStore,
} from "react";

export type AppLang = "tw" | "en" | "ga";

const labels: Record<AppLang, string> = {
  tw: "Twi",
  en: "English",
  ga: "Ga",
};

const LANG_KEY = "gha_lang";
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

function subscribe(onStoreChange: () => void) {
  listeners.add(onStoreChange);
  return () => {
    listeners.delete(onStoreChange);
  };
}

function readLang(): AppLang {
  try {
    const saved = localStorage.getItem(LANG_KEY) as AppLang | null;
    if (saved && labels[saved]) return saved;
  } catch {
    /* SSR / private mode */
  }
  return "tw";
}

function getServerSnapshot(): AppLang {
  return "tw";
}

type Ctx = {
  lang: AppLang;
  setLang: (l: AppLang) => void;
  label: string;
  labels: typeof labels;
};

const LangContext = createContext<Ctx | null>(null);

export function LangProvider({ children }: { children: React.ReactNode }) {
  const lang = useSyncExternalStore(subscribe, readLang, getServerSnapshot);

  const setLang = useCallback((l: AppLang) => {
    try {
      localStorage.setItem(LANG_KEY, l);
    } catch {
      /* ignore */
    }
    emit();
  }, []);

  const value = useMemo(
    () => ({ lang, setLang, label: labels[lang], labels }),
    [lang, setLang],
  );

  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang() {
  const ctx = useContext(LangContext);
  if (!ctx) throw new Error("useLang requires LangProvider");
  return ctx;
}
