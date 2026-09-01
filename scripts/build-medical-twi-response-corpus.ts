import fs from "node:fs/promises";
import path from "node:path";
import "../src/config/load-env";
import { chatComplete, llmProviderInfo } from "../src/lib/llm";

type MedicalSeed = {
  id: string;
  topic: string;
  english_user: string;
  faithful_english_meaning: string;
  intent: string;
  entities: Record<string, string | string[] | boolean>;
  answer: string;
  safety_level: "routine" | "same_day" | "urgent" | "emergency";
  source_urls: string[];
};

type TwiDraft = MedicalSeed & {
  twi_user: string;
  twi_answer: string;
  translation_model: string;
  translation_status: "draft" | "not_requested";
};

const root = process.cwd();
const defaultSourceOut = path.join(root, "data", "medical-response-corpus", "seed.v0.jsonl");
const defaultDraftOut = path.join(root, "tmp", "medical-response-corpus", "twi-drafts.v0.jsonl");

const sourceUrls = {
  maternalCdc: "https://www.cdc.gov/hearher/maternal-warning-signs/index.html",
  maternalWho: "https://www.ncbi.nlm.nih.gov/books/NBK304178/",
  fluCdc: "https://www.cdc.gov/flu/signs-symptoms/index.html",
  malariaWho: "https://www.who.int/news-room/fact-sheets/detail/malaria",
  redEyeNhs: "https://www.nhs.uk/symptoms/red-eye/",
};

const seeds: MedicalSeed[] = [
  {
    id: "med-maternal-headache-vision",
    topic: "pregnancy_warning",
    english_user: "I am pregnant and my headache is getting worse, and my vision is blurry.",
    faithful_english_meaning:
      "Pregnant person reports worsening headache with blurred vision.",
    intent: "report_pregnancy_danger_sign",
    entities: { pregnancy: true, symptom: ["worsening headache", "blurred vision"] },
    answer:
      "Please seek medical care immediately. A worsening headache with vision changes during pregnancy can be a danger sign.",
    safety_level: "urgent",
    source_urls: [sourceUrls.maternalCdc, sourceUrls.maternalWho],
  },
  {
    id: "med-maternal-bleeding",
    topic: "pregnancy_warning",
    english_user: "I am pregnant and I am bleeding like my period.",
    faithful_english_meaning: "Pregnant person reports vaginal bleeding heavier than spotting.",
    intent: "report_pregnancy_bleeding",
    entities: { pregnancy: true, symptom: "vaginal bleeding" },
    answer:
      "Go to a hospital or health centre now. Bleeding during pregnancy needs urgent assessment.",
    safety_level: "emergency",
    source_urls: [sourceUrls.maternalCdc, sourceUrls.maternalWho],
  },
  {
    id: "med-maternal-swollen-face",
    topic: "pregnancy_warning",
    english_user: "My face and hands are suddenly very swollen during pregnancy.",
    faithful_english_meaning: "Pregnant person reports sudden swelling of face and hands.",
    intent: "report_pregnancy_swelling",
    entities: { pregnancy: true, symptom: ["face swelling", "hand swelling"] },
    answer:
      "Please contact a clinic or hospital today. Sudden swelling of the face or hands during pregnancy can be serious.",
    safety_level: "urgent",
    source_urls: [sourceUrls.maternalCdc, sourceUrls.maternalWho],
  },
  {
    id: "med-maternal-breathing",
    topic: "pregnancy_warning",
    english_user: "I am pregnant and I cannot breathe well when I lie down.",
    faithful_english_meaning: "Pregnant person reports trouble breathing while lying flat.",
    intent: "report_breathing_difficulty",
    entities: { pregnancy: true, symptom: "trouble breathing lying down" },
    answer:
      "Seek urgent medical care now, especially if the breathing difficulty is new, worsening, or comes with chest tightness.",
    safety_level: "urgent",
    source_urls: [sourceUrls.maternalCdc, sourceUrls.maternalWho],
  },
  {
    id: "med-maternal-baby-moving-less",
    topic: "pregnancy_warning",
    english_user: "My baby is moving less than usual today.",
    faithful_english_meaning: "Pregnant person reports reduced fetal movement.",
    intent: "report_reduced_fetal_movement",
    entities: { pregnancy: true, symptom: "reduced baby movement" },
    answer:
      "Please contact your maternity clinic or hospital now. A clear change in the baby's movement should be checked.",
    safety_level: "urgent",
    source_urls: [sourceUrls.maternalCdc],
  },
  {
    id: "med-child-fast-breathing",
    topic: "child_fever",
    english_user: "My child has fever and is breathing very fast.",
    faithful_english_meaning: "Child has fever with fast breathing.",
    intent: "report_child_breathing_danger",
    entities: { patient: "child", symptom: ["fever", "fast breathing"] },
    answer:
      "Please seek urgent medical care now. Fever with fast or difficult breathing in a child can be serious.",
    safety_level: "urgent",
    source_urls: [sourceUrls.fluCdc],
  },
  {
    id: "med-child-not-drinking",
    topic: "child_fever",
    english_user: "My child has fever and is not drinking fluids.",
    faithful_english_meaning: "Child has fever with poor fluid intake.",
    intent: "report_child_dehydration_risk",
    entities: { patient: "child", symptom: ["fever", "not drinking"] },
    answer:
      "Try small frequent sips if the child can swallow, and seek medical care today. Go urgently if the child is weak, very sleepy, or not urinating.",
    safety_level: "same_day",
    source_urls: [sourceUrls.fluCdc],
  },
  {
    id: "med-child-seizure",
    topic: "child_fever",
    english_user: "My child had a seizure with fever.",
    faithful_english_meaning: "Child had a seizure while febrile.",
    intent: "report_child_seizure",
    entities: { patient: "child", symptom: ["fever", "seizure"] },
    answer:
      "Get emergency medical help now. A seizure with fever needs urgent assessment.",
    safety_level: "emergency",
    source_urls: [sourceUrls.fluCdc],
  },
  {
    id: "med-malaria-fever-chills",
    topic: "malaria",
    english_user: "I have fever, chills, headache, and I feel very weak.",
    faithful_english_meaning: "Adult reports fever, chills, headache, and weakness suggestive of possible malaria or another infection.",
    intent: "report_possible_malaria_symptoms",
    entities: { symptom: ["fever", "chills", "headache", "weakness"] },
    answer:
      "These symptoms can happen with malaria or other infections. Please get tested and treated by a health worker rather than guessing.",
    safety_level: "same_day",
    source_urls: [sourceUrls.malariaWho],
  },
  {
    id: "med-malaria-dark-urine",
    topic: "malaria",
    english_user: "I have fever and my urine is dark.",
    faithful_english_meaning: "Person reports fever with dark urine.",
    intent: "report_malaria_danger_sign",
    entities: { symptom: ["fever", "dark urine"] },
    answer:
      "Please seek urgent care now. Fever with dark urine can be a danger sign and should be checked quickly.",
    safety_level: "urgent",
    source_urls: [sourceUrls.malariaWho],
  },
  {
    id: "med-malaria-convulsion",
    topic: "malaria",
    english_user: "Someone with fever is having repeated convulsions.",
    faithful_english_meaning: "Person with fever is having multiple convulsions.",
    intent: "report_convulsions_with_fever",
    entities: { symptom: ["fever", "multiple convulsions"] },
    answer:
      "Call emergency help or go to the nearest hospital immediately. Convulsions with fever are an emergency.",
    safety_level: "emergency",
    source_urls: [sourceUrls.malariaWho],
  },
  {
    id: "med-eye-pain-red",
    topic: "eye",
    english_user: "My eye is very red and painful.",
    faithful_english_meaning: "Person reports a very painful red eye.",
    intent: "report_eye_pain_redness",
    entities: { body_part: "eye", symptom: ["redness", "pain"] },
    answer:
      "Please get medical help urgently, especially if the pain is strong, your vision changes, or you wear contact lenses.",
    safety_level: "urgent",
    source_urls: [sourceUrls.redEyeNhs],
  },
  {
    id: "med-eye-discharge-child",
    topic: "eye",
    english_user: "My child's eye is red and has sticky discharge.",
    faithful_english_meaning: "Child has red eye with sticky discharge.",
    intent: "report_child_eye_discharge",
    entities: { patient: "child", body_part: "eye", symptom: ["redness", "sticky discharge"] },
    answer:
      "Keep the eye clean and arrange medical advice, especially if there is pain, swelling, fever, or vision change.",
    safety_level: "same_day",
    source_urls: [sourceUrls.redEyeNhs],
  },
  {
    id: "med-chest-pressure-breathing",
    topic: "chest_breathing",
    english_user: "I have pressure in my chest and I cannot breathe well.",
    faithful_english_meaning: "Person reports chest pressure with breathing difficulty.",
    intent: "report_chest_pain_breathing_difficulty",
    entities: { symptom: ["chest pressure", "breathing difficulty"] },
    answer:
      "Seek emergency medical help now. Chest pressure with breathing difficulty can be serious.",
    safety_level: "emergency",
    source_urls: [sourceUrls.maternalCdc, sourceUrls.fluCdc],
  },
  {
    id: "med-vomiting-dehydration",
    topic: "dehydration",
    english_user: "I keep vomiting and cannot keep water down.",
    faithful_english_meaning: "Person reports repeated vomiting and inability to keep fluids down.",
    intent: "report_dehydration_risk",
    entities: { symptom: ["vomiting", "cannot keep fluids down"] },
    answer:
      "Please seek medical care today. Go urgently if you are confused, dizzy, very weak, or not passing urine.",
    safety_level: "same_day",
    source_urls: [sourceUrls.maternalCdc, sourceUrls.fluCdc],
  },
  {
    id: "med-belly-pain-severe",
    topic: "abdominal_pain",
    english_user: "My stomach pain is severe and it is not going away.",
    faithful_english_meaning: "Person reports severe persistent abdominal pain.",
    intent: "report_severe_abdominal_pain",
    entities: { symptom: "severe persistent abdominal pain" },
    answer:
      "Please get medical care urgently, especially if the pain is sudden, worsening, or comes with fever, vomiting, pregnancy, or bleeding.",
    safety_level: "urgent",
    source_urls: [sourceUrls.maternalCdc, sourceUrls.maternalWho],
  },
];

function argValue(name: string, fallback = "") {
  const exact = process.argv.find((arg) => arg.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  if (index >= 0) return process.argv[index + 1] ?? fallback;
  return fallback;
}

function numericArgValue(name: string, fallback: number) {
  const value = Number(argValue(name, String(fallback)));
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function chunk<T>(rows: T[], size: number) {
  const chunks: T[][] = [];
  for (let index = 0; index < rows.length; index += size) chunks.push(rows.slice(index, index + size));
  return chunks;
}

function parseJsonPayload(value: string) {
  return JSON.parse(value.replace(/^```json\s*/i, "").replace(/\s*```$/i, ""));
}

async function writeJsonl(filePath: string, rows: unknown[]) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
}

async function translateChunk(rows: MedicalSeed[]): Promise<TwiDraft[]> {
  const provider = llmProviderInfo();
  if (!provider) {
    return rows.map((row) => ({
      ...row,
      twi_user: "",
      twi_answer: "",
      translation_model: "none",
      translation_status: "not_requested",
    }));
  }
  const content = await chatComplete(
    [
      {
        role: "system",
        content:
          "You translate Ghana health voice-product corpus rows from English into natural Asante Twi/Akan. Return JSON only. Preserve medical urgency and uncertainty. Do not add diagnosis, dosage, medicine names, prices, or hospital names. Keep the Twi simple enough for voice use.",
      },
      {
        role: "user",
        content: JSON.stringify({
          instruction:
            "For each row, translate english_user into twi_user and answer into twi_answer. Keep id unchanged. Return {rows:[{id,twi_user,twi_answer}]} only.",
          rows,
        }),
      },
    ],
    { temperature: 0, maxTokens: 2400 },
  );
  if (!content) throw new Error("No translation content returned.");
  const parsed = parseJsonPayload(content) as { rows?: Array<{ id?: string; twi_user?: string; twi_answer?: string }> };
  const byId = new Map((parsed.rows ?? []).map((row) => [row.id, row]));
  return rows.map((row) => {
    const translated = byId.get(row.id);
    return {
      ...row,
      twi_user: translated?.twi_user?.trim() ?? "",
      twi_answer: translated?.twi_answer?.trim() ?? "",
      translation_model: `${provider.provider}:${provider.model}`,
      translation_status: translated?.twi_user && translated?.twi_answer ? "draft" : "not_requested",
    };
  });
}

async function main() {
  const sourceOut = argValue("--source-out", defaultSourceOut);
  const draftOut = argValue("--out", defaultDraftOut);
  const limit = numericArgValue("--limit", seeds.length);
  const chunkSize = numericArgValue("--chunk-size", 8);
  const dryRun = process.argv.includes("--dry-run");
  const selected = seeds.slice(0, limit);

  await writeJsonl(sourceOut, selected);

  const drafts: TwiDraft[] = [];
  if (dryRun) {
    drafts.push(
      ...selected.map((row) => ({
        ...row,
        twi_user: "",
        twi_answer: "",
        translation_model: "dry-run",
        translation_status: "not_requested" as const,
      })),
    );
  } else {
    for (const rows of chunk(selected, chunkSize)) {
      drafts.push(...(await translateChunk(rows)));
      console.log(`[medical-twi] translated ${drafts.length}/${selected.length}`);
    }
  }

  await writeJsonl(draftOut, drafts);
  console.log(
    JSON.stringify(
      {
        sourceOut,
        draftOut,
        rows: drafts.length,
        translated: drafts.filter((row) => row.translation_status === "draft").length,
        chunkSize,
      },
      null,
      2,
    ),
  );
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
