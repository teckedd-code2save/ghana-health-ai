import type { Metadata, Viewport } from "next";
import { Shell } from "@/components/shell";
import { LangProvider } from "@/components/lang-provider";
import "./globals.css";

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
  themeColor: "#f8f6ef",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full antialiased">
        <LangProvider>
          <Shell>{children}</Shell>
        </LangProvider>
      </body>
    </html>
  );
}
