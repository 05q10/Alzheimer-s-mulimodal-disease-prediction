# DEMO REPLAY: Transparent Patient Selection Audit

This document records the exact, transparent, and non-cherry-picked selection protocol for the 4 verified Digital Twin demo replay personas, drawn strictly from held-out test subjects in `server/data/splits.json`.

---

## 1. Eligibility Criteria & Partition Guard

All subjects were filtered strictly according to the following criteria:
1. **Partition Isolation**: Subject must belong exclusively to the **Held-Out Test Split** ($N = 456$) in `server/data/splits.json`.
   - **Assertion Result**: Verified strictly disjoint from Training ($N = 2,123$) and Validation ($N = 455$) splits ($0$ subject overlap).
2. **Complete 4-Visit Quad**: Observed visits at months $0$, $6$, $12$, and $24$ (ADNI visit codes `bl`, `m06`, `m12`, `m24` using the `visit_month` timeline with $\pm 1$ month tolerance).
3. **Mandatory Targets & Predictors**: Non-null values for:
   - Mini-Mental State Exam (`MMSE`)
   - Clinical Dementia Rating Global score (`CDGLOBAL`)
   - ADAS-Cog-11 Total Score (`TOTSCORE`)
   - Biological Sex (`PTGENDER` / `sex`)
   - Completed Education Years (`PTEDUCAT` / `education_years`)
   - APOE $\varepsilon4$ Allele Count (`APOE4`)
   - Confirmed Clinical Consensus Diagnosis (`DIAGNOSIS` $\in \{1: \text{CN}, 2: \text{MCI}, 3: \text{AD}\}$)

### Cohort Funnel
- **Total Unified Visits**: $22,168$ visits across $3,034$ subjects
- **Total Test Split Visits**: $3,194$ visits across $456$ subjects
- **Test Subjects with Complete 4-Visit Quad**: **$142$ subjects**

---

## 2. Trajectory Stratification & Seeded Random Draw

The $142$ eligible subjects were partitioned into standard clinical trajectory types and selected via a **seeded pseudo-random draw (`random.seed(42)`)** after sorting candidate IDs deterministically. No subject was selected by prediction error.

| Trajectory Type | Description & Pattern | Eligible Pool ($N$) | Selected ADNI Subject | De-Identified Persona Alias |
|---|---|:---:|:---:|---|
| **Case A** | **Stable Cognitively Normal (CN)**<br>CN at all 4 visits ($[1, 1, 1, 1]$) | $40$ | `022_S_4291` | **Eleanor Vance** |
| **Case B** | **CN Converting to MCI**<br>Baseline CN, progressing to MCI by Month 12/24 | $2$ | `031_S_4218` | **Arthur Pendelton** |
| **Case C** | **MCI Converting to AD**<br>Baseline MCI, progressing to AD by Month 24 | $17$ | `035_S_0997` | **Margaret Sterling** |
| **Case D** | **Stable Amnestic MCI**<br>MCI at all 4 visits ($[2, 2, 2, 2]$) | $59$ | `022_S_4444` | **Harold Finch** |
| *(Alternative)* | *Slowly Declining AD ($[3, 3, 3, 3]$)* | $20$ | — | *(Stable MCI selected for Case D)* |

---

## 3. Selected Replay Patients Profile

### Case A: Eleanor Vance — Stable Cognitively Normal
- **ADNI Subject ID**: `022_S_4291` (Internal only; scrubbed from UI & API)
- **Baseline Demographics**: Age $76$ (rounded), Female (`sex = 1`), $16$ years education, APOE $\varepsilon4 = 0$
- **Longitudinal Record**:
  - Month 0 (bl): Diagnosis = CN, MMSE = 30.0, CDR Global = 0.0, CDR-SB = 0.0, ADAS-11 = 3.0
  - Month 6 (m06): Diagnosis = CN, MMSE = 29.0, CDR Global = 0.0, CDR-SB = 0.0, ADAS-11 = 6.0
  - Month 12 (m12): Diagnosis = CN, MMSE = 29.0, CDR Global = 0.0, CDR-SB = 0.0, ADAS-11 = 2.0
  - Month 24 (m24): Diagnosis = CN, MMSE = 30.0, CDR Global = 0.0, CDR-SB = 0.0, ADAS-11 = 3.3

### Case B: Arthur Pendelton — CN to MCI Converter
- **ADNI Subject ID**: `031_S_4218` (Internal only; scrubbed from UI & API)
- **Baseline Demographics**: Age $81$ (rounded), Male (`sex = 0`), $18$ years education, APOE $\varepsilon4 = 1$
- **Longitudinal Record**:
  - Month 0 (bl): Diagnosis = CN, MMSE = 29.0, CDR Global = 0.0, CDR-SB = 0.0, ADAS-11 = 9.7
  - Month 6 (m06): Diagnosis = CN, MMSE = 28.0, CDR Global = 0.0, CDR-SB = 0.0, ADAS-11 = 6.7
  - Month 12 (m12): Diagnosis = MCI, MMSE = 26.0, CDR Global = 0.0, CDR-SB = 0.0, ADAS-11 = 8.3
  - Month 24 (m24): Diagnosis = MCI, MMSE = 29.0, CDR Global = 0.5, CDR-SB = 1.0, ADAS-11 = 8.3

### Case C: Margaret Sterling — MCI to AD Converter
- **ADNI Subject ID**: `035_S_0997` (Internal only; scrubbed from UI & API)
- **Baseline Demographics**: Age $80$ (rounded), Female (`sex = 1`), $14$ years education, APOE $\varepsilon4 = 2$
- **Longitudinal Record**:
  - Month 0 (bl): Diagnosis = MCI, MMSE = 24.0, CDR Global = 0.5, CDR-SB = 1.5, ADAS-11 = 11.0
  - Month 6 (m06): Diagnosis = MCI, MMSE = 24.0, CDR Global = 0.5, CDR-SB = 1.5, ADAS-11 = 17.0
  - Month 12 (m12): Diagnosis = MCI, MMSE = 24.0, CDR Global = 0.5, CDR-SB = 2.5, ADAS-11 = 15.0
  - Month 24 (m24): Diagnosis = AD, MMSE = 22.0, CDR Global = 0.5, CDR-SB = 3.0, ADAS-11 = 19.3

### Case D: Harold Finch — Stable Amnestic MCI
- **ADNI Subject ID**: `022_S_4444` (Internal only; scrubbed from UI & API)
- **Baseline Demographics**: Age $78$ (rounded), Male (`sex = 0`), $16$ years education, APOE $\varepsilon4 = 1$
- **Longitudinal Record**:
  - Month 0 (bl): Diagnosis = MCI, MMSE = 29.0, CDR Global = 0.5, CDR-SB = 1.5, ADAS-11 = 6.3
  - Month 6 (m06): Diagnosis = MCI, MMSE = 27.0, CDR Global = 0.5, CDR-SB = 1.5, ADAS-11 = 7.0
  - Month 12 (m12): Diagnosis = MCI, MMSE = 29.0, CDR Global = 0.5, CDR-SB = 1.5, ADAS-11 = 4.0
  - Month 24 (m24): Diagnosis = MCI, MMSE = 29.0, CDR Global = 0.5, CDR-SB = 1.0, ADAS-11 = 5.7

---

## 4. Test Set Error Distribution & Patient Percentile Context

To ensure complete scientific transparency, each selected patient's pre-visit prediction errors are benchmarked against the full empirical error distribution of all held-out test anchors ($N = 737$--$1,075$ anchor-horizon pairs):

### Empirical Test Anchor Error Distribution
| Target Metric & Horizon | Anchor Count ($N$) | Mean Test MAE | Median (50th Pct) | 75th Percentile | 90th Percentile |
|---|:---:|:---:|:---:|:---:|:---:|
| **MMSE 6-Month** | $740$ | $1.38$ | $1.00$ | $1.90$ | $3.10$ |
| **MMSE 12-Month** | $1,062$ | $1.46$ | $1.00$ | $2.00$ | $3.30$ |
| **MMSE 24-Month** | $838$ | $1.58$ | $1.00$ | $2.10$ | $3.70$ |
| **CDR Global 6-Month** | $737$ | $0.06$ | $0.00$ | $0.00$ | $0.48$ |
| **CDR Global 12-Month** | $1,075$ | $0.11$ | $0.00$ | $0.00$ | $0.50$ |
| **CDR Global 24-Month** | $864$ | $0.14$ | $0.00$ | $0.18$ | $0.50$ |
| **ADAS-11 6-Month** | $742$ | $3.82$ | $2.88$ | $5.57$ | $8.59$ |
| **ADAS-11 12-Month** | $1,064$ | $3.67$ | $2.49$ | $5.24$ | $8.70$ |
| **ADAS-11 24-Month** | $847$ | $4.26$ | $2.77$ | $6.40$ | $10.00$ |

### Error Percentile Ranking for Selected Replay Cases
- **Case A (Eleanor Vance, Stable CN)**:
  - MMSE Error: $0.90$ at 6m ($42.0\text{th}$ pct), $1.00$ at 12m ($44.6\text{th}$ pct), $0.00$ at 24m ($0.0\text{th}$ pct). Well within typical model performance.
- **Case B (Arthur Pendelton, CN $\to$ MCI Converter)**:
  - MMSE Error: $0.70$ at 6m ($31.9\text{th}$ pct), $2.80$ at 12m (**$84.2\text{th}$ pct**), $0.20$ at 24m ($16.9\text{th}$ pct).
  - CDR Global Error at 24m: $0.50$ (**$86.2\text{th}$ pct**).
  - *Clinical Insight*: The elevated error at 12m reflects the acute conversion event where MMSE abruptly fell by 3 points, showcasing how Model B flags accelerated decline before stabilizing.
- **Case C (Margaret Sterling, MCI $\to$ AD Converter)**:
  - MMSE Error: $0.30$ at 6m ($15.8\text{th}$ pct), $1.00$ at 12m ($44.6\text{th}$ pct), $1.20$ at 24m ($53.3\text{th}$ pct).
  - ADAS-11 Error: $3.80$ at 6m ($60.1\text{th}$ pct), $1.10$ at 12m ($24.2\text{th}$ pct), $3.73$ at 24m ($59.3\text{th}$ pct). Solid tracking of continuous progressive decline.
- **Case D (Harold Finch, Stable MCI)**:
  - MMSE Error: $1.80$ at 6m ($71.8\text{th}$ pct), $0.00$ at 12m ($0.0\text{th}$ pct), $0.30$ at 24m ($21.6\text{th}$ pct). Model reliably captures slow-decline indolence.

---

## 5. Benchmark Hard Case (Resilience Audit)

If a demonstration requires showing a high-error / difficult prognosis scenario:
- **Candidate Subject**: `013_S_1120` (De-identified Alias: **Clara Hughes**, Age $80$, Male, APOE $\varepsilon4=1$, Amnestic MCI)
- **12-Month MMSE Error**: **$6.00$ points** (**$98.5\text{th}$ percentile**)
- **Clinical Feature**: Patient exhibited acute cognitive volatility (MMSE swung from 27 down to 21 within 12 months despite CDR remaining at 0.5), demonstrating the clinical necessity of the 90% predictive interval bounds and reliability flags.
