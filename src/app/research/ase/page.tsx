import { Metadata } from "next";
import { UnderstandingWorkbench } from "@/components/understanding-workbench";
import { CorpusReleaseWorkbench } from "@/components/corpus-release-workbench";

export const metadata: Metadata = {
  title: "Research ASE — Ghana Health",
  robots: {
    index: false,
    follow: false,
  },
};

export default async function ResearchAsePage({ searchParams }: { searchParams: Promise<{ release?: string }> }) {
  const { release } = await searchParams;
  if (process.env.NODE_ENV !== "production" && release && /^[a-zA-Z0-9_-]{1,100}$/.test(release)) {
    return <CorpusReleaseWorkbench release={release} />;
  }
  return <UnderstandingWorkbench />;
}
