"use client";

import { useEffect } from "react";
import { setupOfflineFlush } from "@/lib/offline-queue";

export function SwRegister() {
  useEffect(() => {
    setupOfflineFlush();
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {
        /* ignore */
      });
    }
  }, []);
  return null;
}
