import Link from "next/link";
import { ArrowRight, Mic, MessageCircle, ShoppingBag } from "lucide-react";

export default function HomePage() {
  return (
    <div className="fade-up mx-auto max-w-xl pt-4 md:pt-12">
      <p className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-[var(--teal)]">
        Ghana Health
      </p>
      <h1 className="display text-4xl md:text-[3.25rem]">
        Health guidance
        <br />
        <span className="text-[var(--accent-soft)]">in the language of home.</span>
      </h1>
      <p className="mt-5 max-w-md text-base leading-relaxed text-[var(--fg-muted)] md:text-lg">
        Speak in Twi about pregnancy, symptoms, or the market. Clear next steps — not a doctor.
      </p>

      <div className="mt-9 flex flex-wrap gap-3">
        <Link href="/voice" className="btn btn-primary">
          <Mic className="h-4 w-4" />
          Speak now
          <ArrowRight className="h-4 w-4" />
        </Link>
        <Link href="/chat" className="btn btn-secondary">
          <MessageCircle className="h-4 w-4" />
          Type instead
        </Link>
      </div>

      <div className="mt-14 grid gap-3 sm:grid-cols-3">
        {[
          {
            href: "/voice",
            icon: Mic,
            title: "Voice",
            body: "Talk naturally. We listen, then answer.",
          },
          {
            href: "/chat",
            icon: MessageCircle,
            title: "Chat",
            body: "Type in Twi or English when you prefer.",
          },
          {
            href: "/market",
            icon: ShoppingBag,
            title: "Market",
            body: "Staples and everyday items, ready to order.",
          },
        ].map((card) => (
          <Link
            key={card.href}
            href={card.href}
            className="surface group rounded-[var(--radius)] p-4 transition hover:border-[var(--accent)]/25"
          >
            <card.icon className="mb-3 h-4 w-4 text-[var(--accent)]" />
            <p className="font-medium">{card.title}</p>
            <p className="mt-1 text-sm leading-relaxed text-[var(--fg-muted)]">{card.body}</p>
          </Link>
        ))}
      </div>

      <p className="mt-12 text-sm text-[var(--fg-subtle)]">
        Not medical advice. Emergencies: call{" "}
        <span className="text-[var(--fg-muted)]">112</span> or go to the nearest clinic.
      </p>
    </div>
  );
}
