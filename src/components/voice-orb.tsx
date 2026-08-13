"use client";

import type { CSSProperties } from "react";
import { cn } from "@/lib/utils";

export type OrbMode = "idle" | "listening" | "thinking" | "speaking";

type Props = {
  mode: OrbMode;
  level?: number;
  disabled?: boolean;
  onClick?: () => void;
  label?: string;
  className?: string;
  size?: "md" | "lg";
};

export function VoiceOrb({
  mode,
  level = 0,
  disabled,
  onClick,
  label,
  className,
  size = "md",
}: Props) {
  const energy = Math.max(0.08, Math.min(1, level * 18));
  const speed = `${Math.max(4, 12 - energy * 7).toFixed(2)}s`;

  return (
    <div className={cn("flex flex-col items-center gap-4", className)}>
      <button
        type="button"
        disabled={disabled || mode === "thinking"}
        onClick={onClick}
        aria-label={label || "Speak"}
        className={cn(
          "voice-orb",
          mode === "idle" && "voice-orb--idle",
          mode === "listening" && "voice-orb--listening",
          mode === "thinking" && "voice-orb--thinking",
          mode === "speaking" && "voice-orb--speaking",
          size === "lg" && "voice-orb--lg",
        )}
        style={
          {
            "--orb-energy": energy,
            "--orb-speed": speed,
          } as CSSProperties
        }
      >
        <span className="voice-orb__glow" aria-hidden />
        <span className="voice-orb__ball" aria-hidden>
          <span className="voice-orb__blob voice-orb__blob--one" />
          <span className="voice-orb__blob voice-orb__blob--two" />
          <span className="voice-orb__blob voice-orb__blob--three" />
          <span className="voice-orb__blob voice-orb__blob--four" />
          <span className="voice-orb__metal" />
          <span className="voice-orb__highlight" />
        </span>
      </button>
      {label && (
        <p className="max-w-[16rem] text-center text-sm text-[var(--fg-muted)]">{label}</p>
      )}
    </div>
  );
}

export function modeLabel(mode: OrbMode, vad?: string | null): string {
  if (mode === "listening") {
    if (vad === "speech") return "Hearing you…";
    if (vad === "silence") return "Almost done… tap to send";
    return "Listening…";
  }
  if (mode === "thinking") return "One moment…";
  if (mode === "speaking") return "Speaking… tap to interrupt";
  return "Tap to speak";
}

export function localizedModeLabel(mode: OrbMode, lang: string, vad?: string | null): string {
  if (lang !== "tw") return modeLabel(mode, vad);
  if (mode === "listening") {
    if (vad === "speech") return "Mereyɛ wo tie...";
    if (vad === "silence") return "Ɛreyɛ awie...";
    return "Tie no rekɔ so...";
  }
  if (mode === "thinking") return "Meredwene...";
  if (mode === "speaking") return "Merebua... tap sɛ wopɛ sɛ wotwa mu";
  return "Tap na kasa";
}
