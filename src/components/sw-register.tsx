"use client";

import { useEffect } from "react";
import { setupOfflineFlush } from "@/lib/offline-queue";

export function SwRegister() {
  useEffect(() => {
    setupOfflineFlush();
    if (!("serviceWorker" in navigator)) return;

    if (process.env.NODE_ENV !== "production") {
      navigator.serviceWorker
        .getRegistrations()
        .then((registrations) => Promise.all(registrations.map((r) => r.unregister())))
        .then(() => caches.keys())
        .then((keys) => Promise.all(keys.map((key) => caches.delete(key))))
        .catch(() => {
          /* ignore */
        });
      return;
    }

    navigator.serviceWorker.register("/sw.js").catch(() => {
      /* ignore */
    });
  }, []);

  return null;
}
