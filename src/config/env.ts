import "./load-env";
import { z } from "zod";

const envSchema = z.object({
  DATABASE_URL: z.string().min(1),
  JWT_SECRET: z.string().min(16).default("dev-only-change-me-in-prod-32"),
  NEXT_PUBLIC_APP_URL: z.string().default("http://localhost:3000"),
  NEXT_PUBLIC_API_URL: z.string().default("http://localhost:3000/api"),
  VOICE_MODE: z.enum(["stub", "modal"]).default("stub"),
  MODAL_ASR_URL: z.string().url().optional().or(z.literal("")),
  MODAL_ASR_TOKEN: z.string().optional(),
  HEALTH_DISCLAIMER_ENABLED: z
    .string()
    .optional()
    .transform((v) => v !== "false"),
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
