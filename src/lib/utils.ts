import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatGhs(amount: number | string) {
  const n = typeof amount === "string" ? Number(amount) : amount;
  return `GH₵ ${n.toFixed(2)}`;
}
