import { execFileSync } from "node:child_process";

const DEFAULT_APP = "ap-F3x5vbrsPLQh13kBUTvcgA";
const DEFAULT_CALL = "fc-01KZSWZY0ES2NWC01N9PA8GQP4";

const app = process.env.DONDO_MODAL_APP || DEFAULT_APP;
const call = process.env.DONDO_FUNCTION_CALL || DEFAULT_CALL;
const tail = process.env.DONDO_LOG_TAIL || "120";
const shouldPull = process.argv.includes("--pull");
const shouldLogs = !process.argv.includes("--no-logs");

function run(command: string, args: string[]) {
  console.log(`\n$ ${command} ${args.join(" ")}`);
  execFileSync(command, args, { stdio: "inherit" });
}

function capture(command: string, args: string[]) {
  return execFileSync(command, args, { encoding: "utf8" });
}

function printProgressSummary(logs: string) {
  if (!logs.trim()) {
    console.log("\n[dondo] no function logs yet; app may still be scheduling or starting");
    return;
  }
  const stepMatches = [...logs.matchAll(/(\d+)\/800/g)];
  const lastStep = stepMatches.at(-1)?.[1];
  const lossMatches = [
    ...logs.matchAll(/'loss':\s*([0-9.]+).*?'learning_rate':\s*([0-9.e-]+)/g),
  ];
  const lastLoss = lossMatches.at(-1);
  const evalMatches = [
    ...logs.matchAll(/'eval_wer':\s*([0-9.]+).*?'eval_cer':\s*([0-9.]+)/g),
  ];
  const lastEval = evalMatches.at(-1);
  if (!lastStep && !lastLoss && !lastEval) {
    console.log("\n[dondo] logs visible; waiting for checkpoint/progress metrics");
    return;
  }
  const parts = ["[dondo]"];
  if (lastStep) parts.push(`latest_step=${lastStep}/800`);
  if (lastLoss) {
    parts.push(`loss=${lastLoss[1]}`);
    parts.push(`lr=${lastLoss[2]}`);
  }
  if (lastEval) {
    parts.push(`eval_wer=${(Number(lastEval[1]) * 100).toFixed(2)}%`);
    parts.push(`eval_cer=${(Number(lastEval[2]) * 100).toFixed(2)}%`);
  }
  console.log(`\n${parts.join(" ")}`);
}

run("modal", ["app", "list"]);

if (shouldLogs) {
  const logArgs = [
    "app",
    "logs",
    app,
    "--function-call",
    call,
    "--tail",
    tail,
    "--timestamps",
    "--show-function-call-id",
  ];
  console.log(`\n$ modal ${logArgs.join(" ")}`);
  const logs = capture("modal", logArgs);
  process.stdout.write(logs);
  printProgressSummary(logs);
}

if (shouldPull) {
  run("pnpm", ["eval:asr-results:pull"]);
  run("pnpm", ["eval:asr-results", "--", "--all"]);
}
