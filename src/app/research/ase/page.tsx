import { Metadata } from "next";
import { UnderstandingWorkbench } from "@/components/understanding-workbench";

export const metadata: Metadata = {
  title: "Research ASE — Ghana Health",
  robots: {
    index: false,
    follow: false,
  },
};

export default function ResearchAsePage() {
  return <UnderstandingWorkbench />;
}
