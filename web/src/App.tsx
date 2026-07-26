// HemoSight clinical UI shell. M Sujith Sali, ISE VTU.
// Dark mode default, monospace metrics, i18n (kn/hi/en), offline-first queue.
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { DisclaimerBanner } from "./components/DisclaimerBanner";
import { ConfidenceGauge } from "./components/ConfidenceGauge";
import { CanvasViewer } from "./components/CanvasViewer";
import type { AnalysisResponse } from "./types";

const LANGS = ["kn", "hi", "en"] as const;

// Voice playback of key findings via AI4Bharat IndicTTS backend endpoint.
async function playFindings(analysisId: string, lang: string) {
  const res = await fetch(`/tts?analysis_id=${analysisId}&lang=${lang}`);
  if (!res.ok) return;
  const buf = await res.arrayBuffer();
  await new AudioContext().decodeAudioData(buf).then((b) => {
    const ctx = new AudioContext();
    const src = ctx.createBufferSource();
    src.buffer = b; src.connect(ctx.destination); src.start();
  });
}

export default function App() {
  const { t, i18n } = useTranslation();
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const needsReview =
    !result || result.attention_gate.status === "ATTENTION_MISALIGNMENT";

  return (
    <div style={{ background: "#0b0f19", color: "#e5e7eb", minHeight: "100vh" }}>
      <DisclaimerBanner />
      <header style={{ display: "flex", justifyContent: "space-between", padding: 16 }}>
        <h1 style={{ fontSize: 20 }}>{t("title")}</h1>
        <div>
          {LANGS.map((l) => (
            <button key={l} onClick={() => i18n.changeLanguage(l)} style={{ marginLeft: 6 }}>
              {l.toUpperCase()}
            </button>
          ))}
        </div>
      </header>
      <main style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 16, padding: 16 }}>
        <CanvasViewer />
        <aside style={{ fontFamily: "monospace" }}>
          <ConfidenceGauge
            value={result?.detections[0]?.confidence ?? 0}
            needsReview={needsReview}
          />
          <h3>{t("wbc_differential")}</h3>
          <table>
            <tbody>
              {Object.entries(result?.metrics.wbc_differential ?? {
                Neutrophil: 0, Lymphocyte: 0, Monocyte: 0, Eosinophil: 0, Basophil: 0,
              }).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td style={{ paddingLeft: 12 }}>{v as number}</td></tr>
              ))}
            </tbody>
          </table>
          <p>{t("malaria_flag")}: {String(result?.metrics.parasite_flag ?? false)}</p>
          <button onClick={() => result && playFindings(result.analysis_id, i18n.language)}>
            {t("play_voice")}
          </button>
        </aside>
      </main>
    </div>
  );
}
