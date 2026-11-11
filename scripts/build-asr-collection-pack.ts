import fs from "node:fs/promises";
import path from "node:path";

type Bucket =
  | "health_twi"
  | "commerce_twi"
  | "codeswitch_tw_en"
  | "health_en"
  | "phone_noise";

type Prompt = {
  id: string;
  bucket: Bucket;
  language: "tw" | "en" | "tw-en";
  reference: string;
  speaker_label: string;
  domain_tags: string[];
  recording_tags: string[];
};

const outDir =
  process.env.ASR_COLLECTION_OUT_DIR || path.join(process.cwd(), "tmp", "asr-collection-pack");

const targets: Record<Bucket, number> = {
  health_twi: 100,
  commerce_twi: 50,
  codeswitch_tw_en: 50,
  health_en: 50,
  phone_noise: 50,
};

const seeds: Record<Bucket, Omit<Prompt, "id" | "bucket" | "speaker_label">[]> = {
  health_twi: [
    tw("Me yam yɛ me ya na me ho yɛ hyew", ["symptom", "fever"]),
    tw("Me ti yɛ me ya paa firi anɔpa", ["symptom", "headache"]),
    tw("M'ani yɛ me ya na ɛredane kɔkɔɔ", ["symptom", "eye"]),
    tw("Me ba no ho yɛ hyew anadwo yi", ["child_health", "fever"]),
    tw("Me home yɛ den kakra", ["symptom", "breathing"]),
    tw("Medi aduro yi a, me ho yɛ me basaa", ["medicine", "side_effect"]),
    tw("Me yam retu me na mepɛ ORS ho akwankyerɛ", ["symptom", "ors"]),
    tw("Me yam mu yɛ me ya bere a mididi akyi", ["symptom", "stomach"]),
    tw("Me ho popo na me ho yɛ hyew", ["symptom", "malaria"]),
    tw("Me nyinsɛn na me ho nyɛ me dɛ", ["pregnancy", "symptom"]),
  ],
  commerce_twi: [
    tw("Mepɛ sɛ metɔ tomato ma stew", ["shopping", "produce"]),
    tw("Paracetamol boɔ yɛ sɛn", ["shopping", "medicine"]),
    tw("Hwehwɛ ORS a ɛbɛn me ma me", ["shopping", "medicine"]),
    tw("Mepɛ rice bag baako", ["shopping", "staples"]),
    tw("Soap no wɔ stock mu anaa", ["shopping", "household"]),
    tw("Mepɛ sɛ mokra mosquito net ma me", ["shopping", "health_item"]),
    tw("Tomato kilo mmienu bɛyɛ sɛn", ["shopping", "produce"]),
    tw("Mepɛ sɛ metɔ mako ne onion", ["shopping", "produce"]),
    tw("Fa aduro no brɛ me wɔ Adenta", ["shopping", "delivery"]),
    tw("Hwehwɛ store a ɛwɔ Madina a ɛtɔn ORS", ["shopping", "local_search"]),
  ],
  codeswitch_tw_en: [
    twEn("Mepɛ paracetamol na me ti yɛ me ya", ["medicine", "symptom"]),
    twEn("Me ba no fever no akɔ so since yesterday", ["child_health", "fever"]),
    twEn("Mepɛ oral rehydration salts ma diarrhea", ["medicine", "ors"]),
    twEn("Me stomach pain no yɛ me paa", ["symptom", "stomach"]),
    twEn("Mepɛ delivery for tomatoes wɔ East Legon", ["shopping", "delivery"]),
    twEn("Me eye no yɛ me ya na ɛredane red", ["symptom", "eye"]),
    twEn("Can I take paracetamol sɛ me ho yɛ hyew", ["medicine", "fever"]),
    twEn("Mepɛ mosquito net for my child", ["shopping", "health_item"]),
    twEn("Me breathing no yɛ difficult", ["symptom", "breathing"]),
    twEn("Mepɛ price for rice bag", ["shopping", "price"]),
  ],
  health_en: [
    en("I have fever and stomach pain", ["symptom", "fever"]),
    en("My child has diarrhea since last night", ["child_health", "diarrhea"]),
    en("My eye hurts and it is red", ["symptom", "eye"]),
    en("I am pregnant and I have a severe headache", ["pregnancy", "headache"]),
    en("I feel weak and I am sweating", ["symptom", "weakness"]),
    en("Can oral rehydration salts help with diarrhea", ["medicine", "ors"]),
    en("My chest feels tight when I breathe", ["symptom", "breathing"]),
    en("My baby has fever and is not feeding well", ["child_health", "fever"]),
    en("I took medicine and now I feel dizzy", ["medicine", "side_effect"]),
    en("What should I do if malaria symptoms come back", ["symptom", "malaria"]),
  ],
  phone_noise: [
    tw("Me ba no ho yɛ hyew anadwo yi", ["child_health", "fever"], ["phone", "noise"]),
    tw("M'ani yɛ me ya na ɛyɛ me kusuu", ["symptom", "eye"], ["phone", "noise"]),
    tw("Mepɛ tomato kilo mmienu wɔ Madina", ["shopping", "produce"], ["phone", "noise"]),
    twEn("Me stomach pain no ayɛ worse", ["symptom", "stomach"], ["phone", "noise"]),
    en("My child has fever and chills", ["child_health", "fever"], ["phone", "noise"]),
    tw("Me yam retu me mpɛn pii", ["symptom", "diarrhea"], ["phone", "noise"]),
    tw("Paracetamol wɔ hɔ anaa", ["shopping", "medicine"], ["phone", "noise"]),
    twEn("Mepɛ ORS for my child", ["medicine", "ors"], ["phone", "noise"]),
    tw("Me home yɛ den na me ho yɛ hyew", ["symptom", "breathing"], ["phone", "noise"]),
    en("Can you find oral rehydration salts near me", ["shopping", "medicine"], ["phone", "noise"]),
  ],
};

function tw(reference: string, domainTags: string[], recordingTags = ["phone", "quiet"]) {
  return { language: "tw" as const, reference, domain_tags: domainTags, recording_tags: recordingTags };
}

function twEn(reference: string, domainTags: string[], recordingTags = ["phone", "quiet"]) {
  return { language: "tw-en" as const, reference, domain_tags: domainTags, recording_tags: recordingTags };
}

function en(reference: string, domainTags: string[], recordingTags = ["phone", "quiet"]) {
  return { language: "en" as const, reference, domain_tags: domainTags, recording_tags: recordingTags };
}

function expandPrompts() {
  const prompts: Prompt[] = [];
  for (const bucket of Object.keys(seeds) as Bucket[]) {
    const bucketSeeds = seeds[bucket];
    for (let index = 0; index < targets[bucket]; index++) {
      const seed = bucketSeeds[index % bucketSeeds.length];
      const speakerIndex = Math.floor(index / bucketSeeds.length) + 1;
      const utteranceIndex = (index % bucketSeeds.length) + 1;
      prompts.push({
        id: `${bucket}_sp${String(speakerIndex).padStart(3, "0")}_u${String(utteranceIndex).padStart(4, "0")}`,
        bucket,
        speaker_label: `speaker_${String(speakerIndex).padStart(3, "0")}`,
        ...seed,
      });
    }
  }
  return prompts;
}

function csvEscape(value: string) {
  return `"${value.replace(/"/g, '""')}"`;
}

async function main() {
  const prompts = expandPrompts();
  await fs.mkdir(outDir, { recursive: true });

  const jsonl = prompts
    .map((prompt) =>
      JSON.stringify({
        ...prompt,
        audio_path: `MISSING_AUDIO_${prompt.id}`,
        consent: "internal_eval",
        notes: "Read-aloud collection prompt; attach consented audio before eval/training.",
      }),
    )
    .join("\n");

  const csv = [
    "id,bucket,language,speaker_label,reference,domain_tags,recording_tags",
    ...prompts.map((prompt) =>
      [
        prompt.id,
        prompt.bucket,
        prompt.language,
        prompt.speaker_label,
        prompt.reference,
        prompt.domain_tags.join("|"),
        prompt.recording_tags.join("|"),
      ]
        .map(csvEscape)
        .join(","),
    ),
  ].join("\n");

  const readme = `# ASR Collection Pack

Record each prompt once per speaker on the target device. Keep filenames as:

\`\`\`text
<id>.<wav|webm|m4a|mp3|ogg>
\`\`\`

Examples:

\`\`\`text
health_twi_sp001_u0001.wav
commerce_twi_sp003_u0004.webm
\`\`\`

Do not include names, phone numbers, addresses, or private health stories.
Use speaker labels like \`speaker_001\`, not real names.

After recording, build an audio-ready manifest:

\`\`\`bash
ASR_AUDIO_DIR=/path/to/recordings pnpm asr:attach-audio
\`\`\`

For fast collection on one machine, open \`recorder.html\` in a browser. It records one prompt at a time and downloads each audio file with the correct ID.
`;

  await fs.writeFile(path.join(outDir, "prompts.jsonl"), `${jsonl}\n`, "utf8");
  await fs.writeFile(path.join(outDir, "prompts.csv"), `${csv}\n`, "utf8");
  await fs.writeFile(path.join(outDir, "README.md"), readme, "utf8");
  await fs.writeFile(path.join(outDir, "recorder.html"), recorderHtml(prompts), "utf8");
  console.log(`wrote ${prompts.length} prompts -> ${outDir}`);
}

function recorderHtml(prompts: Prompt[]) {
  const payload = JSON.stringify(prompts).replace(/</g, "\\u003c");
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ghana Health AI ASR Recorder</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f4;
      color: #17211b;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    main {
      width: min(760px, 100%);
      display: grid;
      gap: 24px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: clamp(24px, 4vw, 42px);
      font-weight: 760;
      letter-spacing: 0;
    }
    .meta {
      color: #5a665f;
      font-size: 14px;
    }
    .prompt {
      border: 1px solid #dce3dc;
      background: #ffffff;
      border-radius: 8px;
      padding: clamp(20px, 4vw, 36px);
      box-shadow: 0 18px 45px rgb(30 50 35 / 9%);
      display: grid;
      gap: 18px;
    }
    .reference {
      font-size: clamp(28px, 7vw, 58px);
      line-height: 1.08;
      font-weight: 720;
      letter-spacing: 0;
    }
    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .tag {
      border: 1px solid #d7ded7;
      color: #46534b;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 13px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    button, select, input {
      border: 1px solid #cfd8d1;
      background: #ffffff;
      color: #17211b;
      border-radius: 999px;
      min-height: 44px;
      padding: 0 16px;
      font: inherit;
    }
    button.primary {
      background: #0d6b57;
      border-color: #0d6b57;
      color: white;
    }
    button.danger {
      background: #8d2727;
      border-color: #8d2727;
      color: white;
    }
    button:disabled {
      opacity: 0.45;
    }
    .status {
      min-height: 22px;
      color: #516057;
    }
    .bar {
      height: 8px;
      background: #dce3dc;
      border-radius: 99px;
      overflow: hidden;
    }
    .bar span {
      display: block;
      height: 100%;
      width: 0%;
      background: #0d6b57;
    }
    audio {
      width: 100%;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>ASR Recorder</h1>
        <div class="meta" id="progress"></div>
      </div>
      <select id="bucket"></select>
    </header>
    <section class="prompt">
      <div class="meta" id="id"></div>
      <div class="reference" id="reference"></div>
      <div class="tags" id="tags"></div>
      <div class="bar"><span id="bar"></span></div>
      <audio id="playback" controls hidden></audio>
      <div class="controls">
        <button id="prev">Previous</button>
        <button id="record" class="primary">Record</button>
        <button id="stop" class="danger" disabled>Stop</button>
        <button id="download" disabled>Download</button>
        <button id="next">Next</button>
      </div>
      <div class="status" id="status"></div>
    </section>
  </main>
  <script>
    const prompts = ${payload};
    const bucket = document.getElementById("bucket");
    const progress = document.getElementById("progress");
    const id = document.getElementById("id");
    const reference = document.getElementById("reference");
    const tags = document.getElementById("tags");
    const bar = document.getElementById("bar");
    const status = document.getElementById("status");
    const playback = document.getElementById("playback");
    const prev = document.getElementById("prev");
    const next = document.getElementById("next");
    const record = document.getElementById("record");
    const stop = document.getElementById("stop");
    const download = document.getElementById("download");
    let current = 0;
    let recorder;
    let chunks = [];
    let lastBlob;

    const buckets = ["all", ...Array.from(new Set(prompts.map((p) => p.bucket)))];
    bucket.innerHTML = buckets.map((b) => \`<option value="\${b}">\${b}</option>\`).join("");

    function visiblePrompts() {
      return bucket.value === "all" ? prompts : prompts.filter((p) => p.bucket === bucket.value);
    }

    function render() {
      const list = visiblePrompts();
      if (!list.length) return;
      current = Math.max(0, Math.min(current, list.length - 1));
      const prompt = list[current];
      progress.textContent = \`\${current + 1} of \${list.length}\`;
      id.textContent = \`\${prompt.id} · \${prompt.language} · \${prompt.speaker_label}\`;
      reference.textContent = prompt.reference;
      tags.innerHTML = [...prompt.domain_tags, ...prompt.recording_tags].map((tag) => \`<span class="tag">\${tag}</span>\`).join("");
      bar.style.width = \`\${((current + 1) / list.length) * 100}%\`;
      status.textContent = "Read the prompt naturally. Stop after one clean take.";
      playback.hidden = true;
      playback.removeAttribute("src");
      lastBlob = undefined;
      download.disabled = true;
    }

    async function startRecording() {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
      chunks = [];
      recorder = new MediaRecorder(stream, { mimeType, audioBitsPerSecond: 64000 });
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        lastBlob = new Blob(chunks, { type: mimeType });
        playback.src = URL.createObjectURL(lastBlob);
        playback.hidden = false;
        download.disabled = false;
        status.textContent = "Recorded. Download it, then move to the next prompt.";
      };
      recorder.start();
      record.disabled = true;
      stop.disabled = false;
      status.textContent = "Recording...";
    }

    function stopRecording() {
      if (recorder && recorder.state === "recording") recorder.stop();
      record.disabled = false;
      stop.disabled = true;
    }

    function downloadRecording() {
      if (!lastBlob) return;
      const prompt = visiblePrompts()[current];
      const a = document.createElement("a");
      a.href = URL.createObjectURL(lastBlob);
      a.download = \`\${prompt.id}.webm\`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      status.textContent = \`Downloaded \${a.download}\`;
    }

    bucket.addEventListener("change", () => {
      current = 0;
      render();
    });
    prev.addEventListener("click", () => {
      current -= 1;
      render();
    });
    next.addEventListener("click", () => {
      current += 1;
      render();
    });
    record.addEventListener("click", startRecording);
    stop.addEventListener("click", stopRecording);
    download.addEventListener("click", downloadRecording);
    render();
  </script>
</body>
</html>
`;
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
