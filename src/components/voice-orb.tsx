"use client";

import { Loader2, Mic, Volume2 } from "lucide-react";
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
  const bars = [0.35, 0.55, 0.9, 0.6, 0.4].map((base, i) => {
    if (mode === "listening") {
      return Math.max(4, Math.min(16, base * 12 + level * 120 * (0.7 + i * 0.08)));
    }
    if (mode === "speaking") return 8 + (i % 3) * 3;
    return 5;
  });

  const Icon = mode === "thinking" ? Loader2 : mode === "speaking" ? Volume2 : Mic;

  return (
    <div className={cn("flex flex-col items-center gap-4", className)}>
      <button
        type="button"
        disabled={disabled || mode === "thinking" || mode === "speaking"}
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
      >
        <span className="voice-orb__ring" aria-hidden />
        <span className="voice-orb__ring" aria-hidden />
        <Icon
          className={cn(
            size === "lg" ? "h-11 w-11" : "h-8 w-8",
            mode === "thinking" && "animate-spin",
          )}
          strokeWidth={1.75}
        />
        {(mode === "listening" || mode === "speaking") && (
          <span className="voice-orb__bars" aria-hidden>
            {bars.map((h, i) => (
              <span
                key={i}
                className="voice-orb__bar"
                style={{
                  height: mode === "listening" ? h : undefined,
                  minHeight: 4,
                }}
              />
            ))}
          </span>
        )}
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
    if (vad === "silence") return "Almost done…";
    return "Listening…";
  }
  if (mode === "thinking") return "One moment…";
  if (mode === "speaking") return "Speaking…";
  return "Tap to speak";
}
