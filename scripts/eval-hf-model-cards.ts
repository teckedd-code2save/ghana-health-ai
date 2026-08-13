type RepoExpectation = {
  repo: string;
  baseModel: string;
  datasets: string[];
  metricHints: string[];
};

const repos: RepoExpectation[] = [
  {
    repo: "teckedd/gha-whisper-small-twi-v6",
    baseModel: "openai/whisper-small",
    datasets: ["google/WaxalNLP"],
    metricHints: ["wer", "cer"],
  },
  {
    repo: "teckedd/gha-whisper-small-twi-en-balanced-v7-lite",
    baseModel: "openai/whisper-small",
    datasets: [
      "google/WaxalNLP",
      "fsicoli/common_voice_22_0",
      "ghananlpcommunity/twi-speech-text-multispeaker-16k",
    ],
    metricHints: ["wer", "cer"],
  },
  {
    repo: "teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen",
    baseModel: "openai/whisper-small",
    datasets: [
      "google/WaxalNLP",
      "fsicoli/common_voice_22_0",
      "ghananlpcommunity/twi-speech-text-multispeaker-16k",
    ],
    metricHints: ["wer", "cer"],
  },
  {
    repo: "teckedd/gha-dondo-w2v-bert-twi-v1",
    baseModel: "KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en",
    datasets: ["google/WaxalNLP"],
    metricHints: ["val_wer", "val_cer"],
  },
];

function assertIncludes(haystack: string, needle: string, label: string) {
  if (!haystack.includes(needle)) {
    throw new Error(`${label} missing ${needle}`);
  }
}

async function verifyRepo(expectation: RepoExpectation) {
  const url = `https://huggingface.co/${expectation.repo}/raw/main/README.md`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${expectation.repo} README fetch failed: ${res.status}`);
  }

  const readme = await res.text();
  assertIncludes(readme, "model-index:", expectation.repo);
  assertIncludes(readme, `base_model: ${expectation.baseModel}`, expectation.repo);
  assertIncludes(readme, "datasets:", expectation.repo);
  assertIncludes(readme, "Not a medical device", expectation.repo);
  assertIncludes(readme, "## Intended use", expectation.repo);
  assertIncludes(readme, "## Out of scope", expectation.repo);

  for (const dataset of expectation.datasets) {
    assertIncludes(readme, dataset, expectation.repo);
  }
  for (const metric of expectation.metricHints) {
    assertIncludes(readme.toLowerCase(), metric.toLowerCase(), expectation.repo);
  }

  console.log(`ok hf-card ${expectation.repo}`);
}

async function main() {
  for (const repo of repos) {
    await verifyRepo(repo);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export {};
