import crypto from "node:crypto";

const PAYSTACK_SECRET = process.env.PAYSTACK_SECRET_KEY;
const PAYSTACK_PUBLIC = process.env.PAYSTACK_PUBLIC_KEY;

export function isPaystackConfigured(): boolean {
  return Boolean(PAYSTACK_SECRET);
}

export function getPaystackPublicKey(): string {
  return PAYSTACK_PUBLIC || "";
}

/** Amount in pesewas (GHS * 100) */
export async function initializePayment(input: {
  email: string;
  amount: number;
  reference: string;
  callback_url: string;
  metadata?: Record<string, unknown>;
  /** Ghana mobile money: mtn, vod, atl */
  mobile_money?: { phone: string; provider: "mtn" | "vod" | "atl" };
  channels?: string[];
}) {
  if (!PAYSTACK_SECRET) {
    return { status: false, message: "PAYSTACK_SECRET_KEY not configured" };
  }

  const body: Record<string, unknown> = {
    email: input.email,
    amount: input.amount,
    reference: input.reference,
    callback_url: input.callback_url,
    currency: "GHS",
    metadata: input.metadata,
    channels: input.channels ?? ["mobile_money", "card", "ussd"],
  };

  // For charge API with MoMo we still use initialize + redirect for reliability.
  const response = await fetch("https://api.paystack.co/transaction/initialize", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${PAYSTACK_SECRET}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  return response.json() as Promise<{
    status: boolean;
    message?: string;
    data?: { authorization_url: string; access_code: string; reference: string };
  }>;
}

export type PaystackVerifyResult = {
  status: boolean;
  message?: string;
  data?: {
    status: string;
    reference: string;
    amount: number;
    currency: string;
    customer?: { email?: string };
    metadata?: Record<string, unknown>;
  };
};

export async function verifyPayment(reference: string): Promise<PaystackVerifyResult> {
  if (!PAYSTACK_SECRET) {
    return { status: false, message: "PAYSTACK_SECRET_KEY not configured" };
  }
  const response = await fetch(
    `https://api.paystack.co/transaction/verify/${encodeURIComponent(reference)}`,
    { headers: { Authorization: `Bearer ${PAYSTACK_SECRET}` } },
  );
  return response.json() as Promise<PaystackVerifyResult>;
}

export function verifyWebhookSignature(rawBody: string, signature: string | null): boolean {
  if (!PAYSTACK_SECRET || !signature) return false;
  const hash = crypto.createHmac("sha512", PAYSTACK_SECRET).update(rawBody).digest("hex");
  return hash === signature;
}
