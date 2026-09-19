_Last updated: September 16, 2026_

# Dataset Candidates for a Paired-ECG Design

Candidates for a study that predicts cardiac dysfunction from ECG waveform,
demographics, and metadata. Ranked against one binding requirement.

## The binding requirement

The design needs patients with **at least two ECGs separated by an index event**:
a treatment, a procedure, or a clinical incident. One ECG establishes baseline, the
event defines exposure, the second ECG or a paired echo establishes outcome.

Most public ECG corpora are cross-sectional. They hold one ECG per patient with no
linked timeline, which supports detection ("does this ECG show disease now") but not
prediction ("does this ECG predict disease later"). That single distinction eliminates
most of the field.

A useful first-pass screen is the ratio of ECGs to unique patients. A ratio near 1.0
means the dataset cannot support a paired design at all.

## Screening table

| Dataset | ECGs | Patients | ECGs/patient | Waveforms | Index event linkage | Access |
|---|---|---|---|---|---|---|
| MIMIC-IV-ECG | 800,035 | 161,352 | 4.96 | Yes, 500 Hz | Yes, full EHR | Credentialed |
| EchoNext | 100,000 | 36,286 | 2.76 | Yes, 250 Hz | No | Credentialed |
| ECG-ViEW II | 979,273 | 461,178 | 2.12 | No, see below | Yes, prescriptions | Free, on request |
| CODE-15% | 345,779 | 233,770 | 1.48 | Yes, 400 Hz | No | Open, Zenodo |
| PTB-XL | 21,799 | 18,869 | 1.16 | Yes, 500/100 Hz | No | Open, CC-BY 4.0 |
| Chapman-Shaoxing | 45,152 | ~45,000 | ~1.0 | Yes, 500 Hz | No | Open |
| Ningbo | 34,905 | 40,258 | ~0.87 | Yes, 500 Hz | No | Open |
| SaMi-Trop | 1,631 | 1,959 | ~0.83 | Yes, 400 Hz | Cohort follow-up | Open |

## Tier 1: supports the full design

### MIMIC-IV-ECG plus MIMIC-IV-ECHO plus hosp

The only public resource that supplies all three pieces at once.

- Roughly 5 ECGs per patient, so serial ECGs are the norm rather than the exception.
- [MIMIC-IV-ECHO](https://physionet.org/content/mimic-iv-echo/1.0/) contributes 206,488
  structured echo measurements across 91,372 patients (2008 to 2022), exposing `lvef`,
  `lvef_upper`, `biplane_lvef`, and `lvef_3d`. Serial echos give an LVEF decline endpoint,
  which is the clinical definition of therapy-related cardiac dysfunction.
- The hosp module supplies index events: pharmacy and EMAR drug administration, ICD
  procedure and diagnosis codes, and labs such as troponin and BNP.
- Same institution and era as the ECGs, so patient-level linkage is by `subject_id`.

The open question is attrition. Requiring two echos, a normal baseline, an intervening
ECG, and a specific drug exposure will cut 91,372 patients hard, and no published number
answers how hard. Counting that cohort is the first task, ahead of any model work.

Mitigation if the exposed cohort is too small: define the primary population as incident
LV dysfunction in any patient with a normal baseline echo, and treat drug-exposed
subgroups as secondary. Same design, larger n.

### EchoNext

Not a paired-design dataset on its own, but 2.76 ECGs per patient means serial records
exist, and every ECG carries its own echo-derived labels. That permits a within-dataset
paired endpoint: an ECG whose paired echo is normal, followed by a later ECG whose paired
echo shows `lvef_lte_45_flag`. The transition itself is the incident, with no treatment
information available.

Its stronger role is as the pretraining corpus. Waveforms arrive as N x 1 x 2500 x 12 at
250 Hz with an N x 7 tabular array, which matches the archived two-tower input contract
exactly and needs no resampling. See [archive/ctrcd-ici-study/models.py](archive/ctrcd-ici-study/models.py).

## Tier 2: partial fit

### ECG-ViEW II

Structurally ideal and practically blocked. It holds 979,273 ECGs from 461,178 patients
over 19 years at Ajou University, linked to prescribed drugs, comorbidities, Charlson
index, and electrolytes. Drug exposure plus serial ECGs is exactly the required shape.

The blocker is that ECG-ViEW II distributes **numeric parameters only**: RR, PR, QRS, QT,
QTc, P axis, QRS axis, T axis. There are no raw waveforms, so the waveform tower has
nothing to consume. A separate 2018 publication describes a 12-lead waveform extension;
whether those waveforms are obtainable needs direct confirmation before this dataset can
be ranked higher.

Usable today for a metadata-only and interval-only model, which is a reasonable ablation
arm against the waveform model.

### CODE-15%

At 1.48 ECGs per patient some serial records exist, but there is no index event and no
outcome beyond rhythm and conduction labels. Its value is scale for pretraining at
400 Hz, not study design.

## Tier 3: pretraining and validation only

PTB-XL, Chapman-Shaoxing, and Ningbo are effectively one ECG per patient with no
timeline. PTB-XL remains valuable because it is fully open, needs no credentialing, and
its 500 Hz records decimate cleanly to the 2500-sample contract, making it the right
pipeline smoke test. SaMi-Trop is a genuine cardiomyopathy cohort with study follow-up,
but 1,631 records is too small to anchor a design.

## Candidate index events

Ordered by how many patients each is likely to retain in MIMIC-IV.

| Event | Endpoint | Expected cohort |
|---|---|---|
| Any admission with normal baseline echo | Incident LVEF decline | Largest |
| Cardiac surgery or valve procedure | Post-operative LV dysfunction | Large |
| Myocardial infarction | Post-infarction heart failure | Moderate |
| Sepsis | Septic cardiomyopathy | Moderate |
| Anthracycline administration | Therapy-related dysfunction | Small |
| Immune checkpoint inhibitor administration | Therapy-related dysfunction | Smallest |

The last two preserve the original cardio-oncology question but carry the highest risk of
an unusably small positive class. MIMIC-IV is ICU and ED weighted, so outpatient
oncology infusion is likely under-captured.

## What to verify before committing

1. MIMIC-IV cohort count: patients with two or more echos, LVEF greater than 50 percent
   at baseline, and an ECG between the two studies.
2. How that count falls under each index event in the table above.
3. Whether ECG-ViEW II waveforms are actually obtainable.
4. Positive-class prevalence for `lvef_lte_45_flag` in EchoNext, which is not published
   on the dataset page and determines how useful it is as a pretraining target.

Items 1 and 2 decide the design. Both need only PhysioNet credentialing plus the CITI
"Data or Specimens Only Research" completion report, which clears in roughly a week.

## Technical references

- [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/), 12-lead, 10 s, 500 Hz,
  about 600,000 cardiologist reports. Mirrored free on the AWS Open Data registry.
- [MIMIC-IV-ECHO](https://physionet.org/content/mimic-iv-echo/1.0/), structured
  measurements in `structured_measurement.csv`, study index in `echo-study-list.csv`.
  Cardiologist reports live in MIMIC-IV-Note.
- [EchoNext](https://physionet.org/content/echonext/1.1.1/), DOI 10.13026/7cfw-d091.
  Twelve binary labels including `lvef_lte_45_flag` and `shd_moderate_or_greater_flag`.
  Baseline weights at [github.com/PierreElias/IntroECG](https://github.com/PierreElias/IntroECG)
  under `7-EchoNext Minimodel`.
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), CC-BY 4.0, 10-fold splits with
  folds 9 and 10 human-validated.
- [ECG-ViEW II](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0176222),
  distributed at ecgview.org.
- [CODE-15%](https://doi.org/10.5281/zenodo.4916206).
- Serial-ECG modelling precedent: [deep learning on serial ECGs for emerging cardiac
  pathology](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371549/), covering
  post-infarction heart failure and post-PCI ischemia.
