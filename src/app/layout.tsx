import type { Metadata, Viewport } from "next";
import { DM_Sans, Fraunces } from "next/font/google";
import { Shell } from "@/components/shell";
import { LangProvider } from "@/components/lang-provider";
import { SwRegister } from "@/components/sw-register";
import "./globals.css";

const display = Fraunces({
  variable: "--font-display",
  subsets: ["latin"],
});

const body = DM_Sans({
  variable: "--font-body",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Ghana Health AI — Voice-first health companion",
  description:
    "Twi-first maternal health guidance, voice chat, and market ecommerce for Ghana. Not a substitute for professional care.",
  applicationName: "Ghana Health AI",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Ghana Health AI",
  },
};

export const viewport: Viewport = {
  themeColor: "#0c1a14",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} h-full`}>
      <body className="min-h-full antialiased">
        <LangProvider>
          <SwRegister />
          <Shell>{children}</Shell>
        </LangProvider>
      </body>
    </html>
  );
}
