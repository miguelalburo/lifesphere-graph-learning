"""Every Cypher query the project runs, and the pinned cohort Study list.

Two rules hold throughout this module, and both exist because the schema stores
everything as strings.

**No query casts.** Not one `toInteger` or `toFloat`. The pulls are verbatim, so
`data/raw/` holds exactly what the graph holds, and every cast happens in
`cast.py` where it is covered by tests that do not need the live instance. That
includes the eligibility rule: `timeToEventDays > 0` is a Python predicate in
`cohort.py`, not a Cypher one, so the whole rule sits in one testable place
rather than split across two languages.

**Study membership is the only filter.** It is a string-equality test against
`COHORT_STUDIES`, which needs no cast to be correct.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The cohort's Studies, pinned as a list rather than recomputed from a threshold
# ---------------------------------------------------------------------------

# The 20 Studies where AJCC pathologic staging is present, per the map (#1) and
# verified live on 2026-08-08. Staging coverage is bimodal by Study — 13 Studies
# sit at 0–2.3% and these 20 sit at 85.98–100%, with nothing in between — so
# admitting a Study on "stage unknown" would hand every arm a near-perfect
# cancer-type proxy.
#
# This is a literal list and not a `>= 0.86` filter for a measured reason:
# **TCGA-HNSC's coverage is 0.8598**, so the threshold as written in the map
# yields 19 Studies and silently drops 528 Subjects / 219 events (#4). Pinning
# the members means a change to the graph fails the cohort assertion instead of
# quietly resizing the study.
COHORT_STUDIES: tuple[str, ...] = (
    "TCGA-BLCA",
    "TCGA-BRCA",
    "TCGA-CHOL",
    "TCGA-COAD",
    "TCGA-ESCA",
    "TCGA-HNSC",
    "TCGA-KICH",
    "TCGA-KIRC",
    "TCGA-KIRP",
    "TCGA-LIHC",
    "TCGA-LUAD",
    "TCGA-LUSC",
    "TCGA-MESO",
    "TCGA-PAAD",
    "TCGA-READ",
    "TCGA-SKCM",
    "TCGA-STAD",
    "TCGA-TGCT",
    "TCGA-THCA",
    "TCGA-UVM",
)

# The endpoint this map targets. The property is `survivalType` — *not*
# `endpoint`, which does not exist on the node and returns null for all 37,063
# Survival records, reading as "no records match" rather than as an error.
OS_ENDPOINT = "OS"

# Sample types that exist only because the Subject survived long enough to be
# sampled again. Measured in #4 §4: `Metastatic` scores a *protective* HR of
# 0.749, and the four rarer types have zero deaths between them. Kept out of the
# feature side by `features.py`; pulled verbatim here so `data/raw/` stays a
# faithful copy of the graph.
POST_BASELINE_SAMPLE_TYPES: frozenset[str] = frozenset(
    {
        "Metastatic",
        "Additional - New Primary",
        "Additional Metastatic",
        "Recurrent Tumor",
        "FFPE Scrolls",
    }
)

# The three types a Subject can only have from their presenting workup.
BASELINE_SAMPLE_TYPES: tuple[str, ...] = (
    "Primary Tumor",
    "Blood Derived Normal",
    "Solid Tissue Normal",
)


# ---------------------------------------------------------------------------
# Pulls
# ---------------------------------------------------------------------------

SUBJECTS = """
MATCH (study:Study)-[:HAS_SUBJECT]->(subject:Subject)
WHERE study.studyId IN $studies
RETURN subject.subjectId    AS subjectId,
       study.studyId        AS studyId,
       subject.sexAtBirth   AS sexAtBirth,
       subject.race         AS race,
       subject.ethnicity    AS ethnicity,
       subject.ageAtIndexDays AS ageAtIndexYears
ORDER BY subjectId
"""
"""One row per Subject in the pinned Studies.

`ageAtIndexDays` is aliased to `ageAtIndexYears` at the pull, because the
property **holds years despite its name** — measured range 14–89 in the cohort,
with `-1` used graph-wide as a missing sentinel (#3, #4 §3). Renaming here means
no downstream reader can divide it by 365 on the strength of the property name.
Its sibling `Diagnosis.ageAtDiagnosisDays` genuinely is days.
"""

SURVIVAL = """
MATCH (study:Study)-[:HAS_SUBJECT]->(subject:Subject)-[:HAS_SURVIVAL_RECORD]->(record:Survival)
WHERE study.studyId IN $studies
RETURN record.survivalId      AS survivalId,
       subject.subjectId      AS subjectId,
       study.studyId          AS studyId,
       record.survivalType    AS survivalType,
       record.timeToEventDays AS timeToEventDays,
       record.eventOccurred   AS eventOccurred
ORDER BY subjectId, survivalType
"""
"""Every Survival record for the pinned Studies, all four endpoints.

Endpoint selection is deliberately *not* done here. `cohort.py` filters to OS,
so the raw pull stays endpoint-agnostic and a later PFI/DSS/DFI ticket re-reads
this file rather than the live graph.

All five properties are `STRING NOT NULL` on the live instance, including
`timeToEventDays` and `eventOccurred`. They stay strings through `data/raw/`.
"""

DIAGNOSES = """
MATCH (study:Study)-[:HAS_SUBJECT]->(subject:Subject)
MATCH (subject)-[edge:HAS_DIAGNOSIS]->(diagnosis:Diagnosis)
WHERE study.studyId IN $studies
OPTIONAL MATCH (diagnosis)-[:OF_CONDITION]->(condition:Condition)
RETURN diagnosis.diagnosisId         AS diagnosisId,
       subject.subjectId             AS subjectId,
       edge.isPrimaryDiagnosis       AS isPrimaryDiagnosis,
       diagnosis.pathologicStage     AS pathologicStage,
       diagnosis.ageAtDiagnosisDays  AS ageAtDiagnosisDays,
       diagnosis.conditionSubtype    AS conditionSubtype,
       diagnosis.diagnosisMethod     AS diagnosisMethod,
       condition.conditionId         AS conditionId,
       condition.conditionName       AS conditionName
ORDER BY subjectId, diagnosisId
"""
"""One row per Diagnosis, carrying its edge flag and its Condition inline.

`isPrimaryDiagnosis` is the schema's only edge property (#4 §1) and it holds the
**strings** `'True'` / `'False'`, not booleans. A Cypher `WHERE
edge.isPrimaryDiagnosis = true` matches **zero** rows on this graph — verified
— so it reads as "no primary diagnoses exist" rather than raising. The
comparison is left to `cast.py`, which maps the strings explicitly.

`OF_CONDITION` is an `OPTIONAL MATCH` because ~40% of Diagnoses have no
Condition. That missingness is entirely an artifact of secondary diagnoses: on
the cohort, all 6,808 primary Diagnoses carry a Condition and all 4,867
secondary ones carry none (#4 §1).
"""

SAMPLES = """
MATCH (study:Study)-[:HAS_SUBJECT]->(subject:Subject)
MATCH (subject)-[:PROVIDED_SAMPLE]->(sample:Sample)
WHERE study.studyId IN $studies
RETURN sample.sampleId          AS sampleId,
       subject.subjectId        AS subjectId,
       sample.sampleType        AS sampleType,
       sample.sampleClass       AS sampleClass,
       sample.preservationMethod AS preservationMethod,
       sample.daysToCollection  AS daysToCollection
ORDER BY subjectId, sampleId
"""
"""One row per Sample.

`preservationMethod` and `daysToCollection` are pulled so that `data/raw/`
mirrors the graph, and are dropped at the feature boundary — `daysToCollection`
correlates with follow-up time at r = 0.711, which is to say a sample collected
on day N proves the Subject was alive on day N (#4 §4). `guards.py` asserts
neither reaches a design matrix.
"""

PATHOLOGY_DETAILS = """
MATCH (study:Study)-[:HAS_SUBJECT]->(subject:Subject)
MATCH (subject)-[:HAS_DIAGNOSIS]->(diagnosis:Diagnosis)-[:HAS_PATHOLOGY]->(detail:PathologyDetail)
WHERE study.studyId IN $studies
RETURN detail.pathologyDetailId AS pathologyDetailId,
       diagnosis.diagnosisId    AS diagnosisId,
       subject.subjectId        AS subjectId,
       detail.necrosisPercent   AS necrosisPercent
ORDER BY subjectId, diagnosisId, pathologyDetailId
"""
"""One row per PathologyDetail.

Kept as a featureless structural node: `necrosisPercent` is absent from every
cohort row (all 84 nodes carrying it graph-wide fall outside the cohort), the
node is non-prognostic, and it is not a cancer-type channel (#4 §5). It is
pulled because the graph arm's construction needs the nodes and edges, not
because it contributes a feature.

`HAS_PATHOLOGY` is **not** 1:1 with Diagnosis — 1,734 Diagnoses reach 2–6
details (#4 §5), correcting the encoder doc.
"""

# Node types the feature contract drops outright, recorded so their absence is a
# decision rather than an oversight (#4 §4, §5):
#
#   Intervention           — bare presence is itself an immortal-time leak
#                            (93.5% of Subjects; HR 0.716, p = 8.5e-05), and the
#                            layer is complete-bipartite, so its whole
#                            information content under any permutation-invariant
#                            aggregation was that one contaminated bit. Dropping
#                            it removes 821,184 edges, the graph's largest
#                            relation.
#   PhenotypeObservation   — no properties at all, presence varies, and that
#                            presence is a 0.241 cancer-type channel while
#                            predicting nothing (p = 0.91).
#
# Neither is pulled. A ticket that needs them adds a query here.
DROPPED_NODE_TYPES: tuple[str, ...] = ("Intervention", "PhenotypeObservation")


# Named pulls, in the order `raw.py` runs them.
PULLS: dict[str, str] = {
    "subjects": SUBJECTS,
    "survival": SURVIVAL,
    "diagnoses": DIAGNOSES,
    "samples": SAMPLES,
    "pathology_details": PATHOLOGY_DETAILS,
}
