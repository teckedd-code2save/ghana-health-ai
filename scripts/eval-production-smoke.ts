import { spawnSync } from "node:child_process";

const baseUrl = process.env.EVAL_BASE_URL || "https://ghanahealth.serendepify.com";

const checks = [
  ["product readiness", ["pnpm", "eval:product-readiness:prod"]],
  ["live JSON pipeline", ["pnpm", "eval:live-pipeline"]],
  ["live stream pipeline", ["pnpm", "eval:live-stream"]],
  ["voice feedback loop", ["pnpm", "eval:voice-feedback"]],
  ["commerce confirmation", ["pnpm", "eval:commerce-confirm"]],
] as const;

for (const [label, command] of checks) {
  const [bin, ...args] = command;
  const env = {
    ...process.env,
    EVAL_BASE_URL: baseUrl,
  };
  const result = spawnSync(bin, args, {
    env,
    stdio: "inherit",
    shell: false,
  });

  if (result.status !== 0) {
    throw new Error(`${label} failed with exit code ${result.status ?? "unknown"}`);
  }
}

console.log(`ok production-smoke base=${baseUrl}`);
