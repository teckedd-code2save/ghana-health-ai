import React from "react";
import { createRoot } from "react-dom/client";
import { CorpusReleaseWorkbench } from "../../src/components/corpus-release-workbench";
import "./standalone.css";

declare const CORPUS_RELEASE: string;
createRoot(document.getElementById("root")!).render(
  <CorpusReleaseWorkbench release={CORPUS_RELEASE} endpoint="/research/corpus/api" localSaveLabel="Saved" />
);
