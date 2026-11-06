"use client";

import { useEffect, useState } from "react";

export function useR3Available() {
  const [available, setAvailable] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    void fetch("/api/config?research=r3", { cache: "no-store", signal: controller.signal })
      .then(response => response.ok ? response.json() : null)
      .then(config => { if (!controller.signal.aborted) setAvailable(config?.researchR3?.available === true); })
      .catch(() => {});
    return () => controller.abort();
  }, []);
  return available;
}
