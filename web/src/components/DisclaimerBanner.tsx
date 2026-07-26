// Persistent, non-dismissible disclaimer. M Sujith Sali, ISE VTU.
import { useTranslation } from "react-i18next";

export function DisclaimerBanner() {
  const { t } = useTranslation();
  return (
    <div
      role="alert"
      style={{
        position: "sticky", top: 0, zIndex: 1000, background: "#7f1d1d",
        color: "white", padding: "8px 12px", fontSize: 13, fontWeight: 600,
        textAlign: "center",
      }}
    >
      {t("disclaimer")}
    </div>
  );
}
