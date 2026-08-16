"use client";

/**
 * ASR model A/B store — Whisper v6 (default, production) vs DONDO CTC
 * (research endpoint). Mirrors the lang-provider useSyncExternalStore
 * pattern so nav and voice/chat panels stay in sync without
 * set-state-in-effect lint violations.
 */

import { useSyncExternalStore } from "react";

export type AsrModel = "v6" | "dondo";

const KEY = "gha:asr-model";

const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

function subscribe(onStoreChange: () => void) {
  listeners.add(onStoreChange);
  return () => {
    listeners.delete(onStoreChange);
  };
}

export function readAsrModel(): AsrModel {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "dondo" || saved === "v6") return saved;
  } catch {
    /* SSR / private mode */
  }
  return "v6";
}

function getServerSnapshot(): AsrModel {
  return "v6";
}

export function setAsrModel(next: AsrModel) {
  try {
    localStorage.setItem(KEY, next);
  } catch {
    /* ignore */
  }
  emit();
}

export function useAsrModel(): [AsrModel, (next: AsrModel) => void] {
  const model = useSyncExternalStore(subscribe, readAsrModel, getServerSnapshot);
  return [model, setAsrModel];
}
