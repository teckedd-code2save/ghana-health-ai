import Link from "next/link";
import { ArrowRight, HeartPulse, Mic, ShoppingBag, Shield } from "lucide-react";

export default function HomePage() {
  return (
    <div className="space-y-10">
      <section className="grid items-center gap-8 lg:grid-cols-[1.2fr_0.8fr]">
        <div>
          <p className="mb-3 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-[var(--fg-muted)]">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--teal)]" />
            Live · Twi maternal health + voice market
          </p>
          <h1 className="font-[family-name:var(--font-display)] text-4xl leading-[1.1] tracking-tight md:text-5xl">
            Health guidance in the language of home.
          </h1>
          <p className="mt-4 max-w-xl text-base leading-relaxed text-[var(--fg-muted)] md:text-lg">
            Ghana Health AI is a voice-first companion for maternal health, symptom triage with
            strong disclaimers, and everyday market shopping — built for Twi speakers first, with
            privacy and Ghana Data Protection Act consent at the core.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              href="/chat"
              className="inline-flex items-center gap-2 rounded-full bg-[var(--accent)] px-5 py-3 text-sm font-semibold text-[#1a1400]"
            >
              Start chat <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/voice"
              className="inline-flex items-center gap-2 rounded-full border border-white/15 px-5 py-3 text-sm"
            >
              <Mic className="h-4 w-4" /> Voice
            </Link>
          </div>
        </div>

        <div className="glass relative overflow-hidden rounded-[calc(var(--radius)+6px)] p-6">
          <div className="kente-bar absolute inset-x-0 top-0" />
          <p className="mt-2 text-xs uppercase tracking-[0.2em] text-[var(--fg-muted)]">Live pipeline</p>
          <ol className="mt-4 space-y-3 text-sm">
            {[
              "Mic → Twi Whisper ASR (v6, Modal GPU)",
              "Voice ID enroll / verify from real audio",
              "LLM intent: Health · Market · General",
              "Live product search · cart · MoMo / Paystack",
              "Akan TTS reply + ephemeral transcript",
            ].map((step, i) => (
              <li key={step} className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--teal-deep)] text-xs">
                  {i + 1}
                </span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        {[
          {
            icon: HeartPulse,
            title: "Health companion",
            body: "LLM answers in Twi/English with severity flags and escalation language — not a doctor.",
            href: "/chat",
          },
          {
            icon: Mic,
            title: "Voice ID",
            body: "Enroll and verify with your mic. PCM spectral embedding; login + voice consent required.",
            href: "/voice",
          },
          {
            icon: ShoppingBag,
            title: "Market",
            body: "Search staples & OTC in Twi/English, cart, Mobile Money via Paystack when configured.",
            href: "/market",
          },
        ].map((card) => (
          <Link
            key={card.title}
            href={card.href}
            className="glass rounded-[var(--radius)] p-5 transition hover:border-[var(--accent)]/30"
          >
            <card.icon className="mb-3 h-5 w-5 text-[var(--accent)]" />
            <h2 className="font-[family-name:var(--font-display)] text-xl">{card.title}</h2>
            <p className="mt-2 text-sm leading-relaxed text-[var(--fg-muted)]">{card.body}</p>
          </Link>
        ))}
      </section>

      <section className="glass flex flex-col gap-3 rounded-[var(--radius)] p-5 md:flex-row md:items-center md:justify-between">
        <div className="flex items-start gap-3">
          <Shield className="mt-0.5 h-5 w-5 text-[var(--teal)]" />
          <div>
            <p className="font-medium">Safety & privacy</p>
            <p className="text-sm text-[var(--fg-muted)]">
              Not medical advice. Audio is ephemeral by default. Consent recorded for voice and health
              data under Ghana DPA alignment.
            </p>
          </div>
        </div>
        <Link
          href="/login"
          className="shrink-0 text-sm text-[var(--accent-soft)] underline-offset-4 hover:underline"
        >
          Demo account: demo@ghanahealth.ai
        </Link>
      </section>
    </div>
  );
}
