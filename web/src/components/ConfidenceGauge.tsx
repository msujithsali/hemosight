// Calibrated-confidence gauge with a NEEDS REVIEW state. M Sujith Sali, ISE VTU.
import { useTranslation } from "react-i18next";

export function ConfidenceGauge({ value, needsReview }: { value: number; needsReview: boolean }) {
  const { t } = useTranslation();
  const pct = Math.round(value * 100);
  const color = needsReview ? "#f59e0b" : value > 0.85 ? "#16a34a" : "#eab308";
  return (
    <div style={{ fontFamily: "monospace" }}>
      <div style={{ fontSize: 32, color }}>{pct}%</div>
      <div>{t("confidence")}</div>
      {needsReview && (
        <div style={{ color: "#f59e0b", fontWeight: 700 }}>{t("needs_review")}</div>
      )}
    </div>
  );
}
