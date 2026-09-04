"use client";

import { useSyncExternalStore } from "react";

export type UnderstandingModelMode = "shadow" | "assist" | "assist_v1";

const KEY = "gha:understanding-model-mode";
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

export function readUnderstandingModelMode(): UnderstandingModelMode {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "shadow" || saved === "assist" || saved === "assist_v1") return saved;
  } catch {
    /* SSR / private mode */
  }
  return "shadow";
}

function getServerSnapshot(): UnderstandingModelMode {
  return "shadow";
}

export function setUnderstandingModelMode(next: UnderstandingModelMode) {
  try {
    localStorage.setItem(KEY, next);
  } catch {
    /* ignore */
  }
  emit();
}

export function useUnderstandingModelMode(): [
  UnderstandingModelMode,
  (next: UnderstandingModelMode) => void,
] {
  const mode = useSyncExternalStore(subscribe, readUnderstandingModelMode, getServerSnapshot);
  return [mode, setUnderstandingModelMode];
}
