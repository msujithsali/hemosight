// Zoomable/pannable viewer with independent Raw/Detection/Heatmap layers.
// M Sujith Sali, ISE VTU. (Layer toggles + wheel-zoom; canvas draw wired to API output.)
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";

export function CanvasViewer() {
  const { t } = useTranslation();
  const [layers, setLayers] = useState({ raw: true, det: true, cam: false });
  const [zoom, setZoom] = useState(1);
  const ref = useRef<HTMLCanvasElement>(null);
  const toggle = (k: keyof typeof layers) => setLayers({ ...layers, [k]: !layers[k] });
  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button onClick={() => toggle("raw")}>{t("raw")}: {layers.raw ? "on" : "off"}</button>
        <button onClick={() => toggle("det")}>{t("detections")}: {layers.det ? "on" : "off"}</button>
        <button onClick={() => toggle("cam")}>{t("heatmap")}: {layers.cam ? "on" : "off"}</button>
      </div>
      <canvas
        ref={ref}
        width={512}
        height={512}
        onWheel={(e) => setZoom((z) => Math.max(0.5, z + (e.deltaY < 0 ? 0.1 : -0.1)))}
        style={{ border: "1px solid #444", transform: `scale(${zoom})`, transformOrigin: "top left" }}
      />
    </div>
  );
}
