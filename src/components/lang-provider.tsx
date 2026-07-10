"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

export type AppLang = "tw" | "en" | "ga" | "ee" | "dag";

const labels: Record<AppLang, string> = {
  tw: "Twi",
  en: "English",
  ga: "Ga",
  ee: "Ewe",
  dag: "Dagbani",
};

type Ctx = {
  lang: AppLang;
  setLang: (l: AppLang) => void;
  label: string;
  labels: typeof labels;
};

const LangContext = createContext<Ctx | null>(null);

export function LangProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<AppLang>("tw");

  useEffect(() => {
    const saved = localStorage.getItem("gha_lang") as AppLang | null;
    if (saved && labels[saved]) setLangState(saved);
  }, []);

  const setLang = (l: AppLang) => {
    setLangState(l);
    localStorage.setItem("gha_lang", l);
  };

  const value = useMemo(
    () => ({ lang, setLang, label: labels[lang], labels }),
    [lang],
  );

  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang() {
  const ctx = useContext(LangContext);
  if (!ctx) throw new Error("useLang requires LangProvider");
  return ctx;
}
