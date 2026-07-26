// HemoSight i18n — Kannada, Hindi, English. M Sujith Sali, ISE VTU.
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./locales/en.json";
import kn from "./locales/kn.json";
import hi from "./locales/hi.json";

i18n.use(initReactI18next).init({
  resources: { en: { t: en }, kn: { t: kn }, hi: { t: hi } },
  lng: "kn", fallbackLng: "en", defaultNS: "t",
  interpolation: { escapeValue: false },
});
export default i18n;
