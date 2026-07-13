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
  title: "Ghana Health — Voice health companion",
  description:
    "Talk health in Twi. Maternal guidance, everyday questions, and market shopping for Ghana. Not a substitute for professional care.",
  applicationName: "Ghana Health",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Ghana Health",
  },
};

export const viewport: Viewport = {
  themeColor: "#071510",
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
