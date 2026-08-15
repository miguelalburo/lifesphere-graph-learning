"""Contract tests for the modelling stack pinned in requirements.txt.

These are not unit tests of project code — there is none yet. They pin the
properties of the *dependencies* that the three-model comparison relies on, so an
upgrade that quietly breaks one fails here rather than in a result table.

Nothing here touches the live Neo4j instance (see tests/README.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    import pandas as pd
    from lifelines import CoxPHFitter
    from torch_geometric.data import HeteroData

# Fixed across the module so a failure is reproducible rather than flaky.
SEED = 0


def _synthetic_survival(
    n: int = 200, *, tied: bool = False, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (risk_score, event, time) for a small synthetic cohort.

    With ``tied=True`` the times are integer day-counts, reproducing the tie
    structure of ``Survival.timeToEventDays`` — which is where concordance
    implementations are most likely to disagree.
    """
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    if tied:
        time = rng.integers(1, 25, size=n).astype(float)
        event = rng.random(n) < 0.3
    else:
        time = rng.exponential(scale=np.exp(-score))
        event = rng.random(n) < 0.5
    return score, event, time


# The effect planted in _stratified_cohort, and the sign pattern the decoder is
# expected to recover. Its length fixes the cohort's covariate count.
PLANTED_BETA = np.array([0.8, -0.5, 0.2])


def _stratified_cohort(
    n: int = 400, n_strata: int = 5, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, stratum, event, time) for a synthetic multi-Study cohort.

    Shaped like the locked cohort rather than like a textbook example: each
    stratum carries its own baseline hazard scale, censoring is heavy, and times
    are integer day-counts. The within-stratum event tie multiplicity peaks at
    4, which is the value measured inside COAD (#3 §1) — the loss never sees the
    pooled multiplicity of 9, so testing at the pooled figure would test a
    regime that does not occur.

    Callers unpack four same-length arrays whose order is easy to transpose, so
    every helper below takes them keyword-only.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(PLANTED_BETA)))
    stratum = rng.integers(0, n_strata, size=n)
    # The per-stratum factor is what stratification is meant to absorb into the
    # baseline hazard; without it the strata are interchangeable and the
    # stratified and unstratified objectives would agree trivially.
    latent = rng.exponential(scale=np.exp(-(x @ PLANTED_BETA)) * (1.0 + stratum))
    censor = rng.exponential(scale=2.0 * (1.0 + stratum), size=n)
    time = np.ceil(np.minimum(latent, censor) * 60.0)
    event = latent <= censor
    return x, stratum, event, time


def _cohort_frame(
    *, x: np.ndarray, stratum: np.ndarray, event: np.ndarray, time: np.ndarray
) -> "pd.DataFrame":
    """The long-form frame lifelines consumes, with `study` as the strata column."""
    import pandas as pd

    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    frame["time"] = time
    frame["event"] = event
    frame["study"] = stratum
    return frame


def _lifelines_stratified_fit(
    *,
    x: np.ndarray,
    stratum: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    penalizer: float = 0.0,
) -> "CoxPHFitter":
    """Fit the decoder every model is scored through (#3 §3)."""
    from lifelines import CoxPHFitter

    fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0, strata=["study"])
    fitter.fit(
        _cohort_frame(x=x, stratum=stratum, event=event, time=time),
        duration_col="time",
        event_col="event",
    )
    return fitter


def _torchsurv_stratified_pll(
    *,
    risk: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    stratum: np.ndarray,
    ties_method: str = "efron",
) -> float:
    """Stratified Cox partial log-likelihood, sign-matched to lifelines.

    ``torchsurv`` returns the *negative* partial log-likelihood; lifelines
    reports the partial log-likelihood itself. The negation here is the only
    difference, and getting it wrong would make every comparison below pass for
    the wrong reason.
    """
    from torchsurv.loss.cox import neg_partial_log_likelihood

    return -float(
        neg_partial_log_likelihood(
            torch.tensor(risk, dtype=torch.float64),
            torch.tensor(event),
            torch.tensor(time, dtype=torch.float64),
            ties_method=ties_method,
            reduction="sum",
            strata=torch.tensor(stratum),
        )
    )


class TestStackPresent:
    """Every library the models are built on imports at the pinned version."""

    def test_imports_resolve(self) -> None:
        import lifelines
        import sksurv
        import torch_geometric
        import torchsurv

        assert torch.__version__.startswith("2.13")
        assert torch_geometric.__version__.startswith("2.8")
        assert sksurv.__version__.startswith("0.28")
        assert lifelines.__version__.startswith("0.30")
        # torchsurv is the one library whose version a bump could slide past,
        # and it is the one the objective-agreement guarantee rests on.
        assert torchsurv.__version__.startswith("0.2")

    def test_the_remaining_pinned_libraries_import(self) -> None:
        """#5 asked for a recorded verification that every import works.

        These carry no contract of their own — they either import or the
        environment is broken — so they are checked here rather than given
        tests of their own.
        """
        import matplotlib
        import neo4j
        import pandas
        import scipy
        import sklearn
        import torchmetrics
        from dotenv import load_dotenv

        assert neo4j.__version__.startswith("6.2")
        assert pandas.__version__.startswith("2.3")
        assert scipy.__version__.startswith("1.18")
        assert sklearn.__version__.startswith("1.9")
        assert matplotlib.__version__.startswith("3.11")
        assert torchmetrics.__version__.startswith("1.9")
        assert callable(load_dotenv)

    def test_torch_is_a_cpu_build(self) -> None:
        """This machine has no GPU; a CUDA wheel here means the pin drifted.

        The CUDA wheels are ~2.5GB and pull a driver stack that cannot run,
        so catching the substitution early is worth a test.
        """
        assert torch.__version__.endswith("+cpu")
        assert not torch.cuda.is_available()


class TestConcordanceIsShared:
    """The models must be scored by one implementation, not two that agree roughly.

    Model code never calls these metrics directly — #3 §6 routes all scoring
    through one entry point in ``gl_lifesphere/survival/``. What these tests
    establish is that the entry point is free to use either library, so the
    choice is an implementation detail rather than something a cross-model delta
    could depend on.
    """

    @staticmethod
    def _both(
        score: np.ndarray, event: np.ndarray, time: np.ndarray
    ) -> tuple[float, float]:
        from sksurv.metrics import concordance_index_censored
        from torchsurv.metrics.cindex import ConcordanceIndex

        sks = concordance_index_censored(event, time, score)[0]
        ts = ConcordanceIndex()(
            torch.tensor(score, dtype=torch.float64),
            torch.tensor(event),
            torch.tensor(time, dtype=torch.float64),
        ).item()
        return float(sks), float(ts)

    def test_agree_without_ties(self) -> None:
        sks, ts = self._both(*_synthetic_survival(tied=False))
        assert sks == pytest.approx(ts, abs=1e-9)

    def test_agree_under_heavy_ties(self) -> None:
        """Integer day-counts tie heavily; this is the case that matters here."""
        score, event, time = _synthetic_survival(tied=True, seed=1)
        _, counts = np.unique(time, return_counts=True)
        assert counts.max() >= 10, "fixture is meant to be tie-heavy"

        sks, ts = self._both(score, event, time)
        assert sks == pytest.approx(ts, abs=1e-9)

    def test_uno_ipcw_corrects_harrell_optimism_under_censoring(self) -> None:
        """Harrell's C is optimistic under censoring; Uno's IPCW variant corrects it.

        Asserting only that Uno returns a number in [0, 1] would be vacuous — a
        C-index is in [0, 1] by construction. What has to hold is that the two
        actually *differ* in the direction the metric choice was made for
        (#3 §5 measures Harrell 0.013 above Uno on the real cohort), and that
        the gap is a censoring artefact rather than two unrelated statistics.
        """
        from sksurv.metrics import concordance_index_ipcw
        from sksurv.util import Surv

        x, _, event, time = _stratified_cohort()
        risk = x @ PLANTED_BETA
        # tau spans the whole follow-up, so truncation cannot masquerade as the
        # censoring correction being measured here.
        tau = float(time[event].max())

        def uno(ev: np.ndarray) -> float:
            y = Surv.from_arrays(event=ev, time=time)
            return float(concordance_index_ipcw(y, y, risk, tau=tau)[0])

        assert 0.3 < 1.0 - event.mean() < 0.6, "fixture must be censored"

        # Optimistic, by a margin this study could actually resolve (~0.009
        # here; #3 §5 measures 0.013 on the real cohort against a fold sd of
        # 0.012).
        harrell, _ = self._both(risk, event, time)
        assert harrell - uno(event) > 5e-3

        # Remove censoring and the correction has nothing left to correct: the
        # IPCW weights all collapse to 1 and the two statistics coincide.
        uncensored = np.ones_like(event)
        harrell_uncensored, _ = self._both(risk, uncensored, time)
        assert uno(uncensored) == pytest.approx(harrell_uncensored, abs=1e-3)

    def test_a_stratum_with_no_comparable_pairs_raises(self) -> None:
        """Small Studies produce empty test folds, and sksurv refuses rather than guesses.

        #3 §5 needs this: such strata contribute zero to both numerator and
        denominator of the pair-pooled within-Study C and are listed explicitly
        in the results table. That rule is only implementable because the
        failure is an exception — if sksurv ever returned 0.0 or nan instead,
        the aggregation would silently average a fabricated value in.
        """
        from sksurv.exceptions import NoComparablePairException
        from sksurv.metrics import concordance_index_censored

        # The only event falls after every censoring time, so no pair is comparable.
        event = np.array([False, False, True])
        time = np.array([1.0, 2.0, 9.0])

        with pytest.raises(NoComparablePairException):
            concordance_index_censored(event, time, np.array([0.1, 0.2, 0.3]))


class TestSurvivalLoss:
    """The Cox partial likelihood must be differentiable back into an encoder."""

    def test_cox_loss_is_finite_and_propagates_gradient(self) -> None:
        from torchsurv.loss.cox import neg_partial_log_likelihood

        score, event, time = _synthetic_survival()
        risk = torch.tensor(score, dtype=torch.float32, requires_grad=True)

        loss = neg_partial_log_likelihood(
            risk.unsqueeze(1),
            torch.tensor(event),
            torch.tensor(time, dtype=torch.float32),
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert risk.grad is not None
        assert torch.isfinite(risk.grad).all()
        assert risk.grad.norm() > 0, "a zero gradient would train nothing"

    def test_loss_is_invariant_to_a_constant_shift_of_risk(self) -> None:
        """The intercept is unidentifiable, which is why the head carries no bias.

        See #3 §1, which locks the head to `Linear(d, 1, bias=False)`. If a
        future version of torchsurv ever broke this invariance, a bias term
        would silently become trainable and the head's weights would stop
        being readable.
        """
        from torchsurv.loss.cox import neg_partial_log_likelihood

        score, event, time = _synthetic_survival()
        ev, tm = torch.tensor(event), torch.tensor(time, dtype=torch.float32)

        base = neg_partial_log_likelihood(
            torch.tensor(score, dtype=torch.float32).unsqueeze(1), ev, tm
        )
        shifted = neg_partial_log_likelihood(
            torch.tensor(score + 3.7, dtype=torch.float32).unsqueeze(1), ev, tm
        )
        assert float(base) == pytest.approx(float(shifted), abs=1e-4)


class TestRiskScoreConventions:
    """The risk score's scale is free; its sign is not (#3 §6).

    Convention: higher ``r`` = higher hazard = shorter survival. Models 1-3 emit
    ``r`` from different code, so the properties the shared scorer relies on are
    pinned here rather than assumed.
    """

    @staticmethod
    def _planted_risk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A genuinely discriminating score, so a sign flip is visible."""
        x, _, event, time = _stratified_cohort()
        return x @ PLANTED_BETA, event, time

    @staticmethod
    def _c_index(score: np.ndarray, event: np.ndarray, time: np.ndarray) -> float:
        from sksurv.metrics import concordance_index_censored

        return float(concordance_index_censored(event, time, score)[0])

    def test_concordance_ignores_scale_and_monotone_transforms(self) -> None:
        """Models need not agree on units — only on ordering.

        This is what lets a neural model's raw logit and a Cox model's log partial
        hazard go into the same metric without rescaling.
        """
        risk, event, time = self._planted_risk()
        base = self._c_index(risk, event, time)

        assert self._c_index(2.0 * risk + 5.0, event, time) == pytest.approx(base)
        assert self._c_index(np.exp(risk), event, time) == pytest.approx(base)

    def test_reversing_the_sign_reflects_concordance(self) -> None:
        risk, event, time = self._planted_risk()
        base = self._c_index(risk, event, time)

        assert base > 0.6, "fixture must discriminate, or the flip is invisible"
        assert self._c_index(-risk, event, time) == pytest.approx(1.0 - base)

    def test_lifelines_concordance_index_uses_the_opposite_convention(self) -> None:
        """Pins the trap that ``lifelines.utils.concordance_index`` is banned for.

        It takes a predicted *survival time*, not a risk, so handing it ``r``
        returns ``1 - C`` — roughly 0.29 where the truth is 0.71. That does not
        raise; it reads as a bad model, which is why #3 §6 bans the function in
        model code and routes all scoring through one entry point instead.

        The assertion is deliberately two-sided: it fails both if lifelines ever
        flips to the risk convention (silently un-banning it) and if sksurv
        flips away from it.
        """
        from lifelines.utils import concordance_index

        risk, event, time = self._planted_risk()
        theirs = float(concordance_index(time, risk, event))

        assert theirs == pytest.approx(1.0 - self._c_index(risk, event, time))
        assert float(concordance_index(time, -risk, event)) == pytest.approx(
            self._c_index(risk, event, time)
        )


class TestObjectiveAgreement:
    """torchsurv and lifelines must optimise the *same* stratified Efron objective.

    This is the assumption the whole two-library split rests on (#3 §3):
    ``torchsurv`` supplies the differentiable stratified loss that trains the
    encoder, and ``lifelines.CoxPHFitter(strata=['study'])`` is the single
    decoder every model is refitted through. If the two objectives drift apart, a
    version bump silently puts the three models on different targets and any
    cross-model delta stops being attributable to the representation.
    """

    # Measured at these pins: Efron agrees to ~4e-5 (float32 rounding inside
    # torchsurv), Breslow is off by ~1.3 log-units. The gap between the two
    # bounds is three orders of magnitude, so neither is a knife-edge.
    AGREEMENT_TOL = 1e-3
    DISAGREEMENT_FLOOR = 1e-1

    def test_stratified_efron_agrees_with_lifelines(self) -> None:
        x, stratum, event, time = _stratified_cohort()
        fitter = _lifelines_stratified_fit(x=x, stratum=stratum, event=event, time=time)

        theirs = float(fitter.log_likelihood_)
        ours = _torchsurv_stratified_pll(
            risk=x @ fitter.params_.values, event=event, time=time, stratum=stratum
        )

        assert ours == pytest.approx(theirs, abs=self.AGREEMENT_TOL)

    def test_the_fixture_actually_ties_within_strata(self) -> None:
        """Guards the test above: on untied data Efron and Breslow coincide.

        Without ties the agreement test would pass under either ties_method and
        would therefore pin nothing.
        """
        _, stratum, event, time = _stratified_cohort()
        _, counts = np.unique(
            np.stack([stratum[event], time[event]]), axis=1, return_counts=True
        )

        # 4 is COAD's measured within-Study multiplicity (#3 §1); the seed is fixed.
        assert counts.max() == 4, "fixture is meant to tie within strata"

    def test_breslow_does_not_agree(self) -> None:
        """The agreement is specific to Efron, which is why Efron is locked.

        Efron is not merely the nicer default here — it is the only setting
        under which the training loss and the decoder fit are provably the same
        objective. A future default flip to Breslow must fail loudly.
        """
        x, stratum, event, time = _stratified_cohort()
        fitter = _lifelines_stratified_fit(x=x, stratum=stratum, event=event, time=time)
        risk = x @ fitter.params_.values

        breslow = _torchsurv_stratified_pll(
            risk=risk, event=event, time=time, stratum=stratum, ties_method="breslow"
        )

        assert abs(breslow - float(fitter.log_likelihood_)) > self.DISAGREEMENT_FLOOR

    def test_lifelines_optimum_is_stationary_for_torchsurv(self) -> None:
        """Equal values at one point would not prove equal *surfaces*.

        lifelines cannot be made to evaluate its likelihood at an arbitrary
        coefficient — ``fit_options={'max_steps': 0}`` still moves the
        parameters off ``initial_point`` — so the surfaces are compared through
        the gradient instead: lifelines' optimum must also be a stationary
        point of torchsurv's objective. This is the property the two-stage
        wiring needs, since stage one descends torchsurv's surface and stage two
        reports lifelines' optimum on it.
        """
        from torchsurv.loss.cox import neg_partial_log_likelihood

        x, stratum, event, time = _stratified_cohort()
        fitter = _lifelines_stratified_fit(x=x, stratum=stratum, event=event, time=time)

        def grad_at(beta: np.ndarray) -> float:
            coef = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
            neg_partial_log_likelihood(
                torch.tensor(x, dtype=torch.float64) @ coef,
                torch.tensor(event),
                torch.tensor(time, dtype=torch.float64),
                ties_method="efron",
                reduction="sum",
                strata=torch.tensor(stratum),
            ).backward()
            assert coef.grad is not None
            return float(coef.grad.abs().max())

        beta_hat = fitter.params_.values
        # Measured: ~2e-3 at the optimum against ~70 a short step away.
        assert grad_at(beta_hat) < 0.05
        assert grad_at(beta_hat + 0.3) > 1.0

    def test_penalised_log_likelihood_is_not_the_bare_partial_likelihood(self) -> None:
        """``log_likelihood_`` nets off the penalty once ``penalizer > 0``.

        The decoder is fitted penalised (#3 §3), and λ is selected by comparing
        stratified partial log-likelihoods across folds. Reading
        ``log_likelihood_`` for that comparison would silently compare
        *penalised* objectives, so a λ that shrinks harder would look worse than
        it is and selection would bias toward small λ. Model code must recompute
        the partial likelihood at the fitted coefficients instead — which is
        what the agreement above licenses.
        """
        x, stratum, event, time = _stratified_cohort()
        fitter = _lifelines_stratified_fit(
            x=x, stratum=stratum, event=event, time=time, penalizer=0.1
        )

        bare = _torchsurv_stratified_pll(
            risk=x @ fitter.params_.values, event=event, time=time, stratum=stratum
        )

        # The penalty is subtracted, so the reported number is the smaller one.
        assert float(fitter.log_likelihood_) < bare - 1.0


class TestDecoder:
    """The shared decoder is lifelines, not scikit-survival.

    #3 §3 makes ``lifelines.CoxPHFitter(penalizer=λ, l1_ratio=0,
    strata=['study'])`` the single decoder every model is refitted and scored
    through, and rules out ``sksurv.CoxnetSurvivalAnalysis`` because it has no
    ``strata`` parameter. These tests exercise the decoder that was chosen.
    """

    def test_decoder_recovers_the_sign_of_a_planted_effect(self) -> None:
        """A sign error here is silent downstream (#3 §6), so it is caught at the fit."""
        x, stratum, event, time = _stratified_cohort()
        fitter = _lifelines_stratified_fit(x=x, stratum=stratum, event=event, time=time)

        recovered = fitter.params_.values
        assert np.array_equal(np.sign(recovered), np.sign(PLANTED_BETA))

    def test_the_ruled_out_decoder_really_cannot_stratify(self) -> None:
        """Pins the reason Coxnet was rejected, rather than trusting the note.

        If a future sksurv gained a ``strata`` parameter, #3 §3's rejection
        would be worth revisiting — and this test is what would surface that.
        Adopting Coxnet while it lacks one would silently reverse the
        stratification decision and let the cancer-type confound back into the
        loss.
        """
        import inspect

        from sksurv.linear_model import CoxnetSurvivalAnalysis

        params = inspect.signature(CoxnetSurvivalAnalysis).parameters
        assert "strata" not in params

    def test_decoder_returns_a_baseline_hazard_per_stratum(self) -> None:
        """#3 §3 takes Breslow's per-Study baseline hazards from this same fit.

        They are what the integrated Brier score needs, so a version that
        collapsed them to a single pooled curve would break calibration
        reporting without touching the C-index.
        """
        x, stratum, event, time = _stratified_cohort()
        fitter = _lifelines_stratified_fit(x=x, stratum=stratum, event=event, time=time)

        assert fitter.baseline_cumulative_hazard_.shape[1] == len(np.unique(stratum))

    def test_proportional_hazards_test_reports_a_p_value_per_covariate(self) -> None:
        """#3 §4 runs this and it *fails* for stage — so it must really compute.

        ``assert callable(...)`` would pass against a stub. The PH violation is
        a pre-registered part of the analysis, so the test exercises the call
        and checks the shape of what comes back.
        """
        from lifelines.statistics import proportional_hazard_test

        x, stratum, event, time = _stratified_cohort()
        fitter = _lifelines_stratified_fit(x=x, stratum=stratum, event=event, time=time)

        frame = _cohort_frame(x=x, stratum=stratum, event=event, time=time)

        result = proportional_hazard_test(fitter, frame)
        p_values = result.summary["p"]

        assert len(p_values) == x.shape[1]
        assert p_values.notna().all()
        assert ((p_values >= 0.0) & (p_values <= 1.0)).all()


class TestGraphStack:
    """PyG must handle the heterogeneous, rooted subgraph the encoder consumes."""

    @staticmethod
    def _one_subject_graph(d: int = 8) -> "HeteroData":
        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["Subject"].x = torch.randn(1, d)
        data["Sample"].x = torch.randn(4, d)
        data["Subject", "PROVIDED_SAMPLE", "Sample"].edge_index = torch.tensor(
            [[0, 0, 0, 0], [0, 1, 2, 3]]
        )
        return data

    def test_to_undirected_materialises_reverse_edges(self) -> None:
        """Every LifeSphere relation points away from Subject.

        Without reverse edges the root node receives no messages at all and the
        encoder silently degenerates to a function of the Subject's own
        demographics. This test pins the transform that prevents it.
        """
        import torch_geometric.transforms as T

        data = T.ToUndirected()(self._one_subject_graph())

        assert ("Sample", "rev_PROVIDED_SAMPLE", "Subject") in data.edge_types
        # The root must now be a destination, not only a source.
        rev = data["Sample", "rev_PROVIDED_SAMPLE", "Subject"].edge_index
        assert rev[1].unique().tolist() == [0]

    def test_hetero_message_passing_reaches_the_root(self) -> None:
        import torch_geometric.transforms as T
        from torch_geometric.nn import HeteroConv, SAGEConv

        d = 8
        data = T.ToUndirected()(self._one_subject_graph(d))
        conv = HeteroConv(
            {et: SAGEConv((-1, -1), d) for et in data.edge_types}, aggr="sum"
        )

        out = conv(data.x_dict, data.edge_index_dict)

        assert out["Subject"].shape == (1, d)
        assert torch.isfinite(out["Subject"]).all()

    def test_rgcn_conv_runs_on_a_multi_relation_graph(self) -> None:
        from torch_geometric.nn import RGCNConv

        d = 8
        conv = RGCNConv(d, d, num_relations=2)
        x = torch.randn(5, d)
        edge_index = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 3, 4]])
        edge_type = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        out = conv(x, edge_index, edge_type)

        assert out.shape == (5, d)
        assert torch.isfinite(out).all()
