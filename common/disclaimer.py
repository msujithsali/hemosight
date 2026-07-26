"""Single source of truth for the mandatory screening-aid disclaimer.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Every surface that shows results (React banner, PDF footer, README top,
MODEL_CARD, API response `disclaimer` field) imports the SAME string from
here so the wording can never drift. HemoSight is a *screening aid*, not a
diagnostic device.
"""
from __future__ import annotations

DISCLAIMER: str = (
    "SCREENING AID ONLY — NOT A DIAGNOSTIC DEVICE. HemoSight produces "
    "abnormality flags with calibrated confidence and a needs-review flag. "
    "It does not diagnose disease. Every result must be confirmed by a "
    "qualified pathologist or physician. Results on public benchmarks "
    "(Raabin-WBC, NIH Malaria) are off-domain from Indian PHC settings and "
    "require TARGET-tagged clinical validation before any real-world use."
)

DISCLAIMER_SHORT: str = (
    "Screening aid only — not a diagnostic device. Confirm with a clinician."
)

# Language never allowed in any output: HemoSight emits abnormality flags,
# never a disease name.
FORBIDDEN_DIAGNOSTIC_TERMS = (
    "diagnosed with",
    "you have",
    "patient has",
    "confirmed case of",
    "definitely",
)
