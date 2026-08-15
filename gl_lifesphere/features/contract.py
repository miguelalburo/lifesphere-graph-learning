"""The fitted feature contract: fold-aware fit/transform over `raw.assemble_raw_frame`.

Every statistic that could leak held-out information — a category vocabulary,
an imputation value, a standardisation mean/std — is computed once, on the
training-fold Subjects only, and frozen into a `FeatureContract`. `transform`
then applies those frozen statistics to any Subject set, train or held-out
(#4 §3: "Continuous features are standardised on training-fold statistics
only"; the same rule is applied here to every fitted statistic, not only the
continuous ones, since a category vocabulary or a rare-level threshold learned
from held-out Subjects is the same class of leak).

This is the one place both the tabular model (#10) and the graph model (#12) are
required to call through (#10's own text: "Every feature must come from
`gl_lifesphere/features/` ... any divergence is a bug in the comparison").

**The `Condition` feature keys on `conditionId`, amending #4 §2 (resolved on
#12, 2026-08-09).** §2 locked the feature as `conditionName`, on the premise
that it is a 173-level cancer-type vocabulary. #11's prototype found it is not:
173 Condition nodes carry 173 distinct `conditionId` but only 54 distinct
`conditionName`, and `ICD10:C22.0` — liver, and the Condition of all 366 cohort
TCGA-LIHC Subjects — is labelled "Malignant melanoma, NOS". Keying on the name
merges across anatomy, so the node's feature is keyed on its actual identity
instead. Schema fidelity was chosen over #4's alternative recommendation to
drop the node, on the understanding that this is the *more* severe cancer-type
channel of the two — **R²_study = 0.948** against 0.793 for `conditionSubtype`,
the highest measured anywhere in this project, with 108 of 121 cohort
`conditionId` values sitting in exactly one Study. That flag is carried in
model 3's config and results rather than left here.

The within-Study headline metric is immune to it by construction (#4 §6: a
feature near-constant within a stratum cannot reorder within-Study pairs, and
the stratified loss absorbs it into the baseline hazard). The pooled secondaries
are the exposed numbers and must be read knowing `Condition` ≈ Study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..extract import guards
from .raw import SAMPLE_PROPORTION_COLUMNS

MISSING_LEVEL = "__MISSING__"
RARE_LEVEL = "__RARE__"

# Folded into "other" unconditionally — a fixed rule, not a training-fold
# statistic, so it applies identically regardless of which Subjects a fold
# happens to train on (#4 §3).
_RACE_OTHER = frozenset(
    {"american indian or alaska native", "native hawaiian or other pacific islander"}
)

# `conditionSubtype`: training-fold n < 20 lumped to `__RARE__` (#4 §3, measured
# threshold). `Condition` gets the same machinery with no lumping floor — #4
# keeps it un-lumped ("flagged: ~= a relabelling of Study") but any category
# that never appears in a training fold still needs a defined home at transform
# time, which `__RARE__` supplies either way.
_CONDITION_SUBTYPE_MIN_COUNT = 20
_CONDITION_MIN_COUNT = 1


@dataclass(frozen=True)
class _CategoricalEncoder:
    """A one-hot vocabulary fitted on training-fold values only.

    `fixed_fold` (race's two rare levels) is applied before counting, so it is
    a schema-level rule rather than a per-fold statistic. Everything else is
    driven by training-fold counts, and any value the transform later meets
    that is not in `categories` — including a level a fold's training split
    never saw — falls into `__RARE__` rather than raising or being dropped.
    """

    column: str
    prefix: str
    categories: tuple[str, ...]
    fixed_fold: frozenset[str] = frozenset()

    @classmethod
    def fit(
        cls, series: pd.Series, *, column: str, prefix: str, min_count: int,
        fixed_fold: frozenset[str] = frozenset(),
    ) -> "_CategoricalEncoder":
        folded = series.where(~series.isin(fixed_fold), "other")
        counts = folded.dropna().value_counts()
        kept = tuple(sorted(counts[counts >= min_count].index.tolist()))
        fitted = cls(column=column, prefix=prefix, categories=kept, fixed_fold=fixed_fold)
        fitted._assert_slugs_are_injective()
        return fitted

    @property
    def output_columns(self) -> list[str]:
        return [f"{self.prefix}_{self._slug(level)}" for level in (*self.categories, RARE_LEVEL, MISSING_LEVEL)]

    @staticmethod
    def _slug(level: str) -> str:
        """A column name, with everything a downstream library reads as syntax removed.

        `conditionId` values are ICD-10 codes (`ICD10:C22.0`), and a `:` or a
        `.` in a column name reads as a namespace or an attribute access in the
        formula-style interfaces this design matrix passes through. The
        replacement is deliberately lossy, which is what
        `_assert_slugs_are_injective` guards.
        """
        collapsed = level.strip().lower()
        return "".join(character if character.isalnum() else "_" for character in collapsed)

    def _assert_slugs_are_injective(self) -> None:
        """Raise unless every category still has its own column after slugging.

        The Condition vocabulary keys on `conditionId` precisely because
        `conditionName` merged 173 identities into 54 (#4 §2's amendment). A
        slug collision would re-create that merge one layer further down, and
        silently — two categories would share a column and the second would
        overwrite the first. Cheap to check once per fit; impossible to spot
        afterwards.
        """
        slugs = [self._slug(level) for level in self.categories]
        collided = sorted({slug for slug in slugs if slugs.count(slug) > 1})
        if collided:
            offenders = {
                slug: sorted(c for c in self.categories if self._slug(c) == slug)
                for slug in collided
            }
            raise AssertionError(
                f"{self.column}: distinct categories share a column slug: {offenders}"
            )

    def transform(self, series: pd.Series) -> pd.DataFrame:
        folded = series.where(~series.isin(self.fixed_fold), "other")
        bucketed = folded.where(folded.isin(self.categories) | folded.isna(), RARE_LEVEL)
        bucketed = bucketed.fillna(MISSING_LEVEL)
        levels = [*self.categories, RARE_LEVEL, MISSING_LEVEL]
        dummies = pd.get_dummies(bucketed).reindex(columns=levels, fill_value=False)
        dummies.columns = [f"{self.prefix}_{self._slug(level)}" for level in levels]
        return dummies.astype("float64")


@dataclass(frozen=True)
class _BinaryEncoder:
    """`sexAtBirth` -> one column, training-fold mode fills the rare missing value."""

    column: str
    positive_level: str
    fill_value: float

    @classmethod
    def fit(cls, series: pd.Series, *, column: str, positive_level: str) -> "_BinaryEncoder":
        mode = series.dropna().mode()
        fill = float((mode.iloc[0] == positive_level)) if len(mode) else 0.0
        return cls(column=column, positive_level=positive_level, fill_value=fill)

    @property
    def output_columns(self) -> list[str]:
        return [f"is_{self.positive_level}"]

    def transform(self, series: pd.Series) -> pd.DataFrame:
        encoded = (series == self.positive_level).astype("float64")
        encoded = encoded.where(series.notna(), self.fill_value)
        return pd.DataFrame({self.output_columns[0]: encoded})


@dataclass(frozen=True)
class _NumericEncoder:
    """Training-fold within-Study median imputation, then training-fold standardisation.

    Falls back to the training-fold global median for a (fold, Study) cell
    with no observed value at all — a small Study can have zero non-null
    training rows for a field even though the field is rarely missing overall.
    """

    column: str
    global_median: float
    study_median: dict[str, float]
    mean: float
    std: float

    @classmethod
    def fit(cls, values: pd.Series, study: pd.Series, *, column: str) -> "_NumericEncoder":
        numeric = values.astype("Float64")
        global_median = float(numeric.median(skipna=True))
        by_study = numeric.groupby(study).median()
        study_median = {str(k): float(v) for k, v in by_study.dropna().items()}
        imputed = cls._impute(numeric, study, global_median, study_median)
        mean = float(imputed.mean())
        std = float(imputed.std(ddof=0)) or 1.0
        return cls(
            column=column, global_median=global_median, study_median=study_median,
            mean=mean, std=std,
        )

    @staticmethod
    def _impute(
        values: pd.Series, study: pd.Series, global_median: float, study_median: dict[str, float],
    ) -> pd.Series:
        fallback = study.map(study_median).astype("Float64")
        fallback = fallback.fillna(global_median)
        return values.fillna(fallback).astype("float64")

    @property
    def output_columns(self) -> list[str]:
        return [self.column]

    def transform(self, values: pd.Series, study: pd.Series) -> pd.DataFrame:
        imputed = self._impute(values.astype("Float64"), study, self.global_median, self.study_median)
        standardised = (imputed - self.mean) / self.std
        return pd.DataFrame({self.column: standardised})


@dataclass(frozen=True)
class _ProportionEncoder:
    """Sample-type proportions: training-fold mean fills a Subject with zero baseline Samples.

    Standardised the same way as any other continuous feature (#4 §3's rule is
    general, not stage/age-specific).
    """

    columns: tuple[str, ...]
    fill_values: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "_ProportionEncoder":
        fill_values = {c: float(frame[c].mean(skipna=True)) for c in SAMPLE_PROPORTION_COLUMNS}
        imputed = frame[list(SAMPLE_PROPORTION_COLUMNS)].fillna(pd.Series(fill_values))
        means = {c: float(imputed[c].mean()) for c in SAMPLE_PROPORTION_COLUMNS}
        stds = {c: float(imputed[c].std(ddof=0)) or 1.0 for c in SAMPLE_PROPORTION_COLUMNS}
        return cls(
            columns=SAMPLE_PROPORTION_COLUMNS, fill_values=fill_values, means=means, stds=stds,
        )

    @property
    def output_columns(self) -> list[str]:
        return list(self.columns)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        imputed = frame[list(self.columns)].fillna(pd.Series(self.fill_values))
        standardised = {c: (imputed[c] - self.means[c]) / self.stds[c] for c in self.columns}
        return pd.DataFrame(standardised)


@dataclass(frozen=True)
class FeatureContract:
    """Every fitted encoder, frozen for one fold's training Subject set."""

    sex: _BinaryEncoder
    race: _CategoricalEncoder
    condition_subtype: _CategoricalEncoder
    condition: _CategoricalEncoder
    stage: _NumericEncoder
    age: _NumericEncoder
    sample_proportions: _ProportionEncoder

    def transform(self, raw_frame: pd.DataFrame, subject_ids: frozenset[str] | None = None) -> pd.DataFrame:
        """The final numeric design matrix, indexed by `subjectId`.

        `subject_ids=None` transforms every row of `raw_frame`. Guard-checked
        before returning: no `:Survival`-derived, Study, or immortal-time
        column may reach a design matrix (`extract.guards.check_feature_frame`).
        """
        frame = raw_frame if subject_ids is None else raw_frame[raw_frame["subjectId"].isin(subject_ids)]
        frame = frame.sort_values("subjectId").reset_index(drop=True)
        study = frame["studyId"]

        age_source = frame["ageAtDiagnosisYears"].astype("Float64")
        age_source = age_source.fillna(frame["ageAtIndexYears"].astype("Float64"))

        blocks = [
            self.sex.transform(frame["sexAtBirth"]),
            self.race.transform(frame["race"]),
            self.stage.transform(frame["stageOrdinal"], study),
            self.age.transform(age_source, study),
            self.condition_subtype.transform(frame["conditionSubtype"]),
            self.condition.transform(frame["conditionId"]),
            self.sample_proportions.transform(frame),
        ]
        design = pd.concat(blocks, axis=1)
        design.index = frame["subjectId"].to_numpy()
        design.index.name = "subjectId"

        guards.check_feature_frame(design, where="features.FeatureContract.transform")
        return design

    @property
    def feature_names(self) -> list[str]:
        return [
            *self.sex.output_columns,
            *self.race.output_columns,
            *self.stage.output_columns,
            *self.age.output_columns,
            *self.condition_subtype.output_columns,
            *self.condition.output_columns,
            *self.sample_proportions.output_columns,
        ]


def fit_feature_contract(raw_frame: pd.DataFrame, train_subject_ids: frozenset[str]) -> FeatureContract:
    """Fit every encoder on `train_subject_ids` alone.

    `raw_frame` may hold Subjects outside the training set (val/test rows are
    typically present so a single `raw.assemble_raw_frame()` call can serve a
    whole fold) — this function ignores everything not in `train_subject_ids`.
    """
    train = raw_frame[raw_frame["subjectId"].isin(train_subject_ids)]
    study = train["studyId"]
    age_source = train["ageAtDiagnosisYears"].astype("Float64")
    age_source = age_source.fillna(train["ageAtIndexYears"].astype("Float64"))

    return FeatureContract(
        sex=_BinaryEncoder.fit(train["sexAtBirth"], column="sexAtBirth", positive_level="male"),
        race=_CategoricalEncoder.fit(
            train["race"], column="race", prefix="race", min_count=1, fixed_fold=_RACE_OTHER
        ),
        condition_subtype=_CategoricalEncoder.fit(
            train["conditionSubtype"],
            column="conditionSubtype",
            prefix="subtype",
            min_count=_CONDITION_SUBTYPE_MIN_COUNT,
        ),
        condition=_CategoricalEncoder.fit(
            train["conditionId"],
            column="conditionId",
            prefix="condition",
            min_count=_CONDITION_MIN_COUNT,
        ),
        stage=_NumericEncoder.fit(train["stageOrdinal"], study, column="stageOrdinal"),
        age=_NumericEncoder.fit(age_source, study, column="ageAtDiagnosisYears"),
        sample_proportions=_ProportionEncoder.fit(train),
    )
