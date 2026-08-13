import fs from "node:fs/promises";
import path from "node:path";

type StreamEvent = {
  event: string;
  data: Record<string, unknown>;
};

const baseUrl = (process.env.EVAL_BASE_URL || "http://localhost:3000").replace(/\/$/, "");
const audioPath = process.env.EVAL_VOICE_AUDIO_PATH;
const language = process.env.EVAL_VOICE_LANGUAGE || "tw";
const focus = process.env.EVAL_VOICE_FOCUS || "health";
const speak = process.env.EVAL_VOICE_SPEAK || "false";

function assertOk(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function mimeForAudio(filePath: string) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".wav") return "audio/wav";
  if (ext === ".mp3") return "audio/mpeg";
  if (ext === ".m4a") return "audio/mp4";
  if (ext === ".ogg" || ext === ".oga") return "audio/ogg";
  return "audio/webm";
}

async function readSse(res: Response): Promise<StreamEvent[]> {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("missing response stream");
  const decoder = new TextDecoder();
  let buffer = "";
  const events: StreamEvent[] = [];

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const eventLine = chunk.split("\n").find((line) => line.startsWith("event:"));
      const dataLine = chunk.split("\n").find((line) => line.startsWith("data:"));
      if (!eventLine || !dataLine) continue;
      events.push({
        event: eventLine.slice("event:".length).trim(),
        data: JSON.parse(dataLine.slice("data:".length).trim()) as Record<string, unknown>,
      });
    }
  }

  return events;
}

function eventIndex(events: StreamEvent[], event: string) {
  return events.findIndex((item) => item.event === event);
}

function stageIndex(events: StreamEvent[], name: string) {
  return events.findIndex((item) => item.event === "stage" && item.data.name === name);
}

function assertOrdered(
  events: StreamEvent[],
  earlier: number,
  later: number,
  message: string,
) {
  assertOk(earlier >= 0, `${message}: missing earlier event`);
  assertOk(later >= 0, `${message}: missing later event`);
  assertOk(earlier < later, message);
}

async function buildForm() {
  const form = new FormData();
  form.append("language", language);
  form.append("focus", focus);
  form.append("speak", speak);

  if (audioPath) {
    const bytes = await fs.readFile(audioPath);
    form.append(
      "audio",
      new Blob([new Uint8Array(bytes)], { type: mimeForAudio(audioPath) }),
      path.basename(audioPath),
    );
  }

  return form;
}

function assertMissingAudioContract(events: StreamEvent[]) {
  const error = events.find((event) => event.event === "error");
  assertOk(error, "voice stream did not emit structured error event");
  assertOk(
    typeof error.data.error === "string" && error.data.error.length > 0,
    "voice stream error event missing message",
  );
  console.log(`ok voice-stream-contract event=error status=${error.data.status ?? "n/a"}`);
}

function assertRealAudioContract(events: StreamEvent[]) {
  const accepted = stageIndex(events, "accepted");
  const asrStarted = stageIndex(events, "asr_started");
  const asr = eventIndex(events, "asr");
  const asrFinal = stageIndex(events, "asr_final");
  const understanding = stageIndex(events, "understanding");
  const assistant = stageIndex(events, "assistant_message");
  const replyDelta = eventIndex(events, "reply_delta");
  const final = eventIndex(events, "final");

  assertOrdered(events, accepted, asrStarted, "accepted must arrive before ASR starts");
  assertOrdered(events, asrStarted, asr, "ASR result must arrive after ASR starts");
  assertOrdered(events, asr, asrFinal, "ASR final stage must arrive after ASR result");
  assertOrdered(events, asrFinal, understanding, "understanding must start after ASR final");
  assertOrdered(events, understanding, assistant, "assistant stage must follow understanding");
  assertOrdered(events, assistant, replyDelta, "reply chunks must follow assistant stage");
  assertOrdered(events, replyDelta, final, "final must arrive after streamed reply chunks");

  const finalData = events[final]?.data;
  assertOk(typeof finalData?.conversationId === "string", "final missing conversationId");
  assertOk(
    typeof finalData?.message === "object" && finalData.message !== null,
    "final missing assistant message",
  );
  console.log(
    `ok voice-stream-contract real-audio stages=${events.length} final=${finalData.conversationId}`,
  );
}

async function main() {
  const form = await buildForm();

  const res = await fetch(`${baseUrl}/api/voice/converse/stream`, {
    method: "POST",
    body: form,
  });
  assertOk(res.ok, `voice stream HTTP ${res.status}`);
  assertOk(
    res.headers.get("content-type")?.includes("text/event-stream"),
    "voice stream did not return SSE content type",
  );

  const events = await readSse(res);
  if (audioPath) assertRealAudioContract(events);
  else assertMissingAudioContract(events);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

export {};
