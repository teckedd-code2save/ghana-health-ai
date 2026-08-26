import { z } from "zod";

const envSchema = z.object({
  DATABASE_URL: z.string().min(1),
  JWT_SECRET: z.string().min(16).default("dev-only-change-me-in-prod-32"),
  NEXT_PUBLIC_APP_URL: z.string().default("http://localhost:3000"),
  NEXT_PUBLIC_API_URL: z.string().default("http://localhost:3000/api"),
  VOICE_MODE: z.enum(["stub", "modal"]).default("modal"),
  MODAL_ASR_URL: z.string().optional().or(z.literal("")),
  MODAL_ASR_TOKEN: z.string().optional(),
  MODAL_ASR_EN_URL: z.string().optional().or(z.literal("")),
  MODAL_ASR_EN_TOKEN: z.string().optional(),
  MODAL_TTS_URL: z.string().optional().or(z.literal("")),
  MODAL_TTS_TOKEN: z.string().optional(),
  MODAL_EMBED_URL: z.string().optional().or(z.literal("")),
  MODAL_EMBED_TOKEN: z.string().optional(),
  EMBED_MODEL: z.string().optional(),
  GROQ_API_KEY: z.string().optional(),
  OPENAI_API_KEY: z.string().optional(),
  OPENAI_BASE_URL: z.string().optional(),
  LLM_MODEL: z.string().optional(),
  OPENAI_LANGUAGE_MODEL: z.string().optional(),
  PAYSTACK_SECRET_KEY: z.string().optional(),
  PAYSTACK_PUBLIC_KEY: z.string().optional(),
  NEXT_PUBLIC_PAYSTACK_PUBLIC_KEY: z.string().optional(),
  GOOGLE_CLIENT_ID: z.string().optional(),
  GOOGLE_CLIENT_SECRET: z.string().optional(),
  RESEND_API_KEY: z.string().optional(),
  EMAIL_FROM: z.string().optional(),
  HEALTH_ESCALATION_HOTLINE: z.string().optional(),
  SENTRY_DSN: z.string().optional(),
  HEALTH_DISCLAIMER_ENABLED: z
    .string()
    .optional()
    .transform((v) => v !== "false"),
  RESEARCH_REVIEW_ENABLED: z
    .string()
    .optional()
    .transform((v) => v === "true"),
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
});

export type AppEnv = z.infer<typeof envSchema>;

let cached: AppEnv | null = null;

export function getEnv(): AppEnv {
  if (cached) return cached;
  const parsed = envSchema.safeParse(process.env);
  if (!parsed.success) {
    const msg = parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ");
    throw new Error(`Invalid environment: ${msg}`);
  }
  cached = parsed.data;
  return cached;
}
