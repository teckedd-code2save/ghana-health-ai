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
        "Sɛ wo hu mogya a ɛsen, ti yare a ɛmu yɛ den, ahonhon, anaa wo nte wo ba a ɔwɔ yafunu mu a, kɔ hospital ntɛm. Gye ANC nhyiamu so.",
      bodyEn:
        "If you notice heavy bleeding, severe headache, swelling of face/hands, or reduced fetal movement, go to a hospital immediately. Keep all ANC appointments.",
      category: "maternal",
      tags: ["pregnancy", "anc", "danger", "nyinsen", "bleeding"],
      source: "GHS / WHO maternal guidelines (summarized)",
    },
    {
      slug: "nutrition-pregnancy",
      titleTw: "Aduane a ɛho hia wɔ nyinsen mu",
      titleEn: "Nutrition in pregnancy",
      bodyTw:
        "Di aduane a ɛwɔ nneɛma pii: nnuaba, nkyene nsuo nnuane, protein (ɛmo, nam, beans), iron tablets sɛ oduruyɛfoɔ de ma wo. Nom nsuo pii.",
      bodyEn:
        "Eat diverse foods: fruits, vegetables, proteins (eggs, fish, beans), and take iron/folate as prescribed. Stay hydrated.",
      category: "maternal",
      tags: ["nutrition", "pregnancy", "aduane", "iron"],
      source: "GAIN maternal health Q&A (summarized)",
    },
    {
      slug: "fever-basic",
      titleTw: "Afe ho akwankyerɛ",
      titleEn: "Basic fever guidance",
      bodyTw:
        "Nom nsuo, hom, na sɛ afe no kɔ soro anaa ɛkɔ so nna pii a, kɔ clinic. Mma wo mfa aduro a wonnim.",
      bodyEn:
        "Rest, hydrate, and seek clinic care if fever is high or lasts several days. Do not take unknown medicines.",
      category: "general",
      tags: ["fever", "afe", "malaria"],
      source: "General community health guidance",
    },
    {
      slug: "postpartum-warning",
      titleTw: "Awo akyi nsɛnkyerɛnne",
      titleEn: "Postpartum warning signs",
      bodyTw:
        "Sɛ mogya sen dodo, ɔyare, anaa wo nte yie a, kɔ oduruyɛfoɔ hɔ. Fa wo ba kɔ weighing / immunization.",
      bodyEn:
        "Heavy bleeding, fever, or feeling very unwell after birth needs urgent care. Take the baby for weighing and immunizations.",
      category: "maternal",
      tags: ["postpartum", "awo", "bleeding"],
      source: "GHS postpartum care (summarized)",
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

  console.log("Seed complete:", {
    demoUser: demo.email,
    password: "demo1234",
    articles: articles.length,
    products: products.length,
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
