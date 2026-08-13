import "../src/config/load-env";
import { PrismaClient } from "@prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";
import bcrypt from "bcryptjs";

const connectionString = process.env.DATABASE_URL;
if (!connectionString) {
  throw new Error("DATABASE_URL required for seed");
}

const adapter = new PrismaPg({ connectionString });
const prisma = new PrismaClient({ adapter });

async function main() {
  const passwordHash = await bcrypt.hash("demo1234", 10);

  const demo = await prisma.user.upsert({
    where: { email: "demo@ghanahealth.ai" },
    update: {},
    create: {
      email: "demo@ghanahealth.ai",
      phone: "+233200000001",
      displayName: "Ama Mensah",
      passwordHash,
      preferredLang: "tw",
      consentVoice: true,
      consentHealth: true,
      role: "user",
    },
  });

  await prisma.consentRecord.createMany({
    data: [
      { userId: demo.id, kind: "voice", granted: true, version: "1.0" },
      { userId: demo.id, kind: "health", granted: true, version: "1.0" },
    ],
    skipDuplicates: true,
  });

  const articles = [
    {
      slug: "anc-danger-signs",
      titleTw: "Nyinsen mu nsɛnkyerɛnne a ɛyɛ hu",
      titleEn: "Danger signs in pregnancy",
      bodyTw:
        "Sɛ wo hu mogya a ɛsen, ti yare a ɛmu yɛ den, ahonhon wɔ anim anaa nsa, anaa wo nte wo ba a ɔwɔ yafunu mu a, kɔ hospital ntɛm. Mfa nna ho. Kɔ antenatal care nhyiamu a wɔahyɛ no nyinaa so.",
      bodyEn:
        "If you notice heavy bleeding, severe headache, swelling of face/hands, or reduced fetal movement, go to a hospital immediately. Do not wait overnight. Keep all antenatal care appointments.",
      category: "maternal",
      tags: ["pregnancy", "anc", "danger", "nyinsen", "bleeding", "mogya", "ti"],
      source: "GHS / WHO maternal guidelines (summarized)",
    },
    {
      slug: "nutrition-pregnancy",
      titleTw: "Aduane a ɛho hia wɔ nyinsen mu",
      titleEn: "Nutrition in pregnancy",
      bodyTw:
        "Di aduane a ɛwɔ nneɛma pii: nnuaba, nkyene nsuo nnuane, protein (ɛmo, nam, beans), iron tablets sɛ oduruyɛfoɔ de ma wo. Nom nsuo pii. Mma nsa nni.",
      bodyEn:
        "Eat diverse foods: fruits, vegetables, proteins (eggs, fish, beans), and take iron/folate as prescribed. Stay hydrated. Avoid alcohol.",
      category: "maternal",
      tags: ["nutrition", "pregnancy", "aduane", "iron", "nyinsen"],
      source: "GAIN maternal health Q&A (summarized)",
    },
    {
      slug: "fever-basic",
      titleTw: "Afe ho akwankyerɛ",
      titleEn: "Basic fever guidance",
      bodyTw:
        "Nom nsuo, hom, na sɛ afe no kɔ soro anaa ɛkɔ so nna pii a, kɔ clinic. Mma wo mfa aduro a wonnim. Wɔ Ghana, afe betumi ayɛ malaria — ma wɔhwɛ wo ntɛm sɛ ɛyɛ den.",
      bodyEn:
        "Rest, hydrate, and seek clinic care if fever is high or lasts several days. Do not take unknown medicines. In Ghana, fever can be malaria — get checked promptly if it is severe.",
      category: "general",
      tags: ["fever", "afe", "malaria", "yare"],
      source: "General community health guidance",
    },
    {
      slug: "postpartum-warning",
      titleTw: "Awo akyi nsɛnkyerɛnne",
      titleEn: "Postpartum warning signs",
      bodyTw:
        "Sɛ mogya sen dodo, afe, anaa wo nte yie a, kɔ oduruyɛfoɔ hɔ. Fa wo ba kɔ weighing ne immunization. Community health worker betumi aboa wo wɔ fie.",
      bodyEn:
        "Heavy bleeding, fever, or feeling very unwell after birth needs urgent care. Take the baby for weighing and immunizations. A community health worker can support you at home.",
      category: "maternal",
      tags: ["postpartum", "awo", "bleeding", "mogya", "baby"],
      source: "GHS postpartum care (summarized)",
    },
    {
      slug: "malaria-pregnancy",
      titleTw: "Malaria ne nyinsen",
      titleEn: "Malaria in pregnancy",
      bodyTw:
        "Sɛ wo wɔ nyinsen mu na wo wɔ afe, ahometew, anaa wo yafunu yɛ wo ya a, kɔ clinic ntɛm. Ma wɔyɛ malaria test. Fa mosquito net da. Mma wo mfa aduro a oduruyɛfoɔ amma wo.",
      bodyEn:
        "If you are pregnant and have fever, chills, or abdominal pain, go to a clinic promptly for a malaria test. Sleep under a mosquito net. Do not take unprescribed medicines.",
      category: "maternal",
      tags: ["malaria", "pregnancy", "nyinsen", "afe", "fever", "net"],
      source: "GHS malaria in pregnancy (summarized)",
    },
    {
      slug: "breastfeeding-start",
      titleTw: "Nufu a wɔde ma akokoaa",
      titleEn: "Starting breastfeeding",
      bodyTw:
        "Fa nufu ma wo ba ntɛm sɛ ɛbɛtumi. Nufu nkoaa yɛ ade pa mfe a edi kan. Sɛ wo yɛ den anaa mogya kɔ so a, kɔ clinic. Bisa community health worker sɛ ɛhia boa.",
      bodyEn:
        "Start breastfeeding as soon as possible after birth. Exclusive breastfeeding is recommended for the first months. If you feel very unwell or bleeding continues, go to a clinic. Ask a community health worker for support.",
      category: "maternal",
      tags: ["breastfeeding", "nufu", "baby", "awo", "postpartum"],
      source: "WHO / GHS infant feeding (summarized)",
    },
    {
      slug: "dehydration-ors",
      titleTw: "Nsukyenee ne ORS",
      titleEn: "Dehydration and ORS",
      bodyTw:
        "Sɛ wo yare a ɛma wo tutu anaa wo fe a, nom nsuo ne ORS sɛ wonya. Sɛ ɛyɛ den, wo yɛ mmerɛw, anaa wo nte yie a, kɔ clinic — titiriw sɛ wo wɔ nyinsen mu.",
      bodyEn:
        "If you have diarrhoea or vomiting, drink fluids and ORS if available. If symptoms are severe, you are weak, or you feel very unwell — especially in pregnancy — go to a clinic.",
      category: "general",
      tags: ["dehydration", "ors", "diarrhea", "vomiting", "nsu"],
      source: "Community ORS guidance (summarized)",
    },
    {
      slug: "headache-pregnancy",
      titleTw: "Ti yaw wɔ nyinsen mu",
      titleEn: "Headache in pregnancy",
      bodyTw:
        "Ti yaw ketewa betumi aba. Nanso sɛ ti yaw no yɛ den, ɛne ahonhon, anaa wo nte yie a, ɛyɛ danger sign — kɔ hospital ntɛm. Mma wo mfa aduro pii a wonnim.",
      bodyEn:
        "Mild headaches can occur. But severe headache with swelling or feeling very unwell can be a danger sign — go to hospital promptly. Do not take many unknown medicines.",
      category: "maternal",
      tags: ["headache", "ti", "pregnancy", "nyinsen", "danger"],
      source: "GHS maternal danger signs (summarized)",
    },
    {
      slug: "when-to-clinic",
      titleTw: "Bere a ɛsɛ sɛ wokɔ clinic",
      titleEn: "When to go to the clinic",
      bodyTw:
        "Kɔ clinic sɛ: afe kɔ so, ɛyɛ den, wo nte yie, wo wɔ nyinsen mu na biribi yɛ wo ya, anaa wo ho yɛ wo anika. Sɛ ɛyɛ emergency a, frɛ 112 anaa kɔ hospital.",
      bodyEn:
        "Go to a clinic if fever persists, pain is severe, you feel very unwell, you are pregnant and something hurts, or something feels wrong. For emergencies call 112 or go to hospital.",
      category: "general",
      tags: ["clinic", "hospital", "112", "emergency", "kɔ"],
      source: "Community care pathways (summarized)",
    },
  ];

  for (const a of articles) {
    await prisma.knowledgeArticle.upsert({
      where: { slug: a.slug },
      update: a,
      create: a,
    });
  }

  const products = [
    {
      sku: "RICE-5KG",
      nameTw: "Ɛmo (kg 5)",
      nameEn: "Rice 5kg",
      descriptionTw: "Ɛmo pa a wɔtɔn wɔ gua so",
      descriptionEn: "Local staple rice, 5kg bag",
      category: "staples",
      priceGhs: 85.0,
      unit: "bag",
      stock: 40,
      tags: ["rice", "ɛmo", "food"],
    },
    {
      sku: "PARA-500",
      nameTw: "Paracetamol 500mg",
      nameEn: "Paracetamol 500mg",
      descriptionTw: "Aduro a ɛte afe / ya (OTC)",
      descriptionEn: "OTC pain and fever relief",
      category: "otc-meds",
      priceGhs: 12.5,
      unit: "pack",
      stock: 100,
      tags: ["paracetamol", "medicine", "fever"],
    },
    {
      sku: "SOAP-KEY",
      nameTw: "Sapo Key",
      nameEn: "Key Soap",
      descriptionTw: "Sapo a wɔde hohoro",
      descriptionEn: "Household laundry soap",
      category: "household",
      priceGhs: 8.0,
      unit: "bar",
      stock: 200,
      tags: ["soap", "sapo", "household"],
    },
    {
      sku: "OIL-1L",
      nameTw: "Ngo (1L)",
      nameEn: "Cooking oil 1L",
      descriptionEn: "Vegetable cooking oil",
      descriptionTw: "Ngo a wɔde noa aduane",
      category: "staples",
      priceGhs: 32.0,
      unit: "bottle",
      stock: 60,
      tags: ["oil", "ngo", "food"],
    },
    {
      sku: "ORS-PACK",
      nameTw: "ORS nsuo aduro",
      nameEn: "ORS sachets",
      descriptionEn: "Oral rehydration salts",
      descriptionTw: "Nsuo a ɛboa sɛ yare a ɛma nsuo tew",
      category: "otc-meds",
      priceGhs: 6.0,
      unit: "pack",
      stock: 80,
      tags: ["ors", "dehydration", "medicine"],
    },
    {
      sku: "MOSQ-NET",
      nameTw: "Ntoma a ɛbɔ mosquitos ho ban",
      nameEn: "Mosquito net",
      descriptionEn: "Treated bed net",
      descriptionTw: "Ntoma a wɔde da de bɔ malaria ho ban",
      category: "household",
      priceGhs: 45.0,
      unit: "piece",
      stock: 25,
      tags: ["net", "malaria", "mosquito"],
    },
  ];

  for (const p of products) {
    await prisma.product.upsert({
      where: { sku: p.sku },
      update: p,
      create: p,
    });
  }

  // Promote demo to admin for audit UI later if needed
  await prisma.user.update({
    where: { id: demo.id },
    data: { role: "admin" },
  });

  const moreArticles = [
    {
      slug: "malaria-basics",
      titleTw: "Malaria ho nsɛm",
      titleEn: "Malaria basics",
      bodyTw:
        "Sɛ wo wɔ afe, tipae, anaa ahonhon a, kɔ clinic ma wɔnhwɛ malaria. Da wɔ ntoma a ɛbɔ mosquito ho ban ase. Yi nsuo a ɛgyina.",
      bodyEn:
        "Fever, chills, or headache may be malaria — get tested at a clinic. Sleep under a treated net and remove standing water.",
      category: "general",
      tags: ["malaria", "fever", "afe", "mosquito"],
      source: "Community health summary",
    },
    {
      slug: "mental-health-triage",
      titleTw: "Adwene mu yare ho",
      titleEn: "Mental health triage",
      bodyTw:
        "Sɛ wo adwenem nte yie anaa wo pɛ sɛ wobɛhaw wo ho a, ka kyerɛ obi a wo gye di anaa frɛ helpline. Yɛnka sɛ yɛyɛ oduruyɛfoɔ.",
      bodyEn:
        "If you feel hopeless or unsafe with yourself, tell a trusted person or call a helpline. This app is not a crisis service.",
      category: "mental",
      tags: ["mental", "stress", "suicide", "helpline"],
      source: "Safety-first triage notes",
    },
    {
      slug: "ors-dehydration",
      titleTw: "Nsuo a ɛtew / ORS",
      titleEn: "Dehydration and ORS",
      bodyTw:
        "Sɛ ɛyare a ɛma nsuo tew (diarrhea) a, nom ORS anaa nsuo a wɔde nkyene ne asikyire ayɛ. Sɛ mmɔfra anaa mmea a wɔnyinsen a, kɔ clinic ntɛm.",
      bodyEn:
        "For diarrhea, use ORS or clean fluids with salt/sugar as guided. Infants and pregnant people should seek clinic care early.",
      category: "general",
      tags: ["ors", "diarrhea", "dehydration", "nsuo"],
      source: "CHW hydration guidance",
    },
    {
      slug: "child-fever",
      titleTw: "Afe wɔ mmɔfra so",
      titleEn: "Fever in children",
      bodyTw:
        "Sɛ abofra wɔ afe a, nom nsuo, hom, na hwɛ ne ho. Sɛ ɛyɛ den, abofra no yɛ mmerɛw, anaa afe no nte a, kɔ clinic ntɛm — titiriw sɛ ɔyɛ ketewa.",
      bodyEn:
        "If a child has fever, offer fluids, rest, and watch closely. If fever is high, the child is weak, or it does not improve — go to a clinic promptly, especially for infants.",
      category: "child",
      tags: ["child", "fever", "afe", "abofra", "mmɔfra"],
      source: "Community pediatric triage (summarized)",
    },
    {
      slug: "anc-visits",
      titleTw: "Antenatal care nhyiamu",
      titleEn: "Antenatal care visits",
      bodyTw:
        "Sɛ wo wɔ nyinsen mu a, kɔ antenatal care nhyiamu a wɔahyɛ no. Wɔbɛhwɛ wo mogya, wo ba a ɔwɔ yafunu mu, ne wo apɔwmuden. Sɛ wo nte yie a, kɔ clinic a ɛnyɛ sɛ ɛsɛ sɛ woda kɔ.",
      bodyEn:
        "If you are pregnant, keep all antenatal care visits. Clinics check blood, fetal wellbeing, and your health. If you feel unwell, go sooner — do not wait for the next appointment.",
      category: "maternal",
      tags: ["anc", "antenatal", "nyinsen", "pregnancy", "clinic"],
      source: "GHS ANC pathway (summarized)",
    },
    {
      slug: "family-planning-referral",
      titleTw: "Awo nhyehyɛeɛ",
      titleEn: "Family planning referral",
      bodyTw:
        "Sɛ wopɛ awo nhyehyɛeɛ (family planning) a, kɔ clinic anaa community health worker hɔ. Wɔbɛkyerɛ wo kwan a ɛfata wo. Yɛnka sɛ yɛbɛma wo aduro foforo wɔ ha.",
      bodyEn:
        "For family planning options, visit a clinic or community health worker. They will explain methods that fit you. This app does not prescribe contraceptives.",
      category: "maternal",
      tags: ["family planning", "awo", "contraception", "clinic"],
      source: "Community FP referral (summarized)",
    },
    {
      slug: "hypertension-diabetes-simple",
      titleTw: "Mogya a ɛkɔ soro ne diabetes",
      titleEn: "High blood pressure and diabetes basics",
      bodyTw:
        "Sɛ wɔaka akyerɛ wo sɛ wo mogya kɔ soro anaa wo wɔ diabetes a, di oduruyɛfoɔ akwankyerɛ so, di aduane a ɛho hia, na kɔ check-up. Sɛ ti yare den, home yɛ den, anaa wo nte yie a, kɔ hospital.",
      bodyEn:
        "If you have high blood pressure or diabetes, follow clinic advice, eat carefully, and keep check-ups. Severe headache, breathing trouble, or feeling very unwell needs urgent care.",
      category: "chronic",
      tags: ["hypertension", "diabetes", "mogya", "pressure", "chronic"],
      source: "Community NCD basics (summarized)",
    },
    {
      slug: "immunization-basics",
      titleTw: "Nnwoma a wɔde bɔ mmɔfra ho ban",
      titleEn: "Childhood immunization basics",
      bodyTw:
        "Fa wo ba kɔ immunization / weighing sɛ ɛsɛ. Nnwoma no bɔ yare bi te sɛ measles ho ban. Sɛ wo werɛ fi da a, kɔ clinic ma wɔnhwɛ schedule no.",
      bodyEn:
        "Take children for immunizations and weighing as scheduled. Vaccines protect against serious diseases such as measles. If you miss a date, visit the clinic to catch up.",
      category: "child",
      tags: ["immunization", "vaccine", "mmɔfra", "weighing", "measles"],
      source: "EPI / community immunization (summarized)",
    },
  ];
  for (const a of moreArticles) {
    await prisma.knowledgeArticle.upsert({
      where: { slug: a.slug },
      update: a,
      create: a,
    });
  }

  console.log("Seed complete:", {
    demoUser: demo.email,
    password: "demo1234",
    role: "admin",
    articles: articles.length + moreArticles.length,
    products: products.length,
    note: "Run: sec -- pnpm db:index-embeddings  (after MODAL_EMBED_URL is set)",
  });
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
