"""Model-comparison statistics.

Ranking these models by R² alone is misleading, because more parameters almost
always buys a better in-sample fit. This module adds the corrections that make
the comparison honest, and an out-of-sample backtest that sidesteps the problem
entirely.

The central difficulty is that the residuals are enormously autocorrelated.
These are daily observations of a slow trend: price stays on one side of the
fitted curve for years at a time, so lag-1 autocorrelation runs above 0.99 and
Durbin–Watson sits near 0.01 instead of 2. Textbook AIC/BIC assume independent
observations, and with n = 5,834 the complexity penalty (2k, or ln(n)·k ≈ 8.7k)
is swamped by a log-likelihood built from thousands of points that carry only a
few dozen observations' worth of independent information. Plain AIC therefore
reproduces the R² ranking exactly and penalises nothing — measured ΔAIC between
the 6-parameter halving-cycle model and the 2-parameter power law is over 2,300,
which is not a real preference.

The fix used here is to discount the likelihood to an *effective* sample size
derived from the residual decorrelation time. See ``effective_sample_size``.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# The ACF is summed until it decays past this level; beyond it the estimate is
# dominated by noise.
ACF_CUTOFF = 0.05

# Walk-forward defaults. The first origin leaves ~4.5 years of training data,
# and origins are spaced half a year apart.
BACKTEST_START = pd.Timestamp("2015-01-01")
BACKTEST_SPACING_DAYS = 182
BACKTEST_HORIZON_DAYS = 365


@dataclass
class Metrics:
    """In-sample fit quality, complexity penalties and residual diagnostics."""

    n: int
    n_params: int
    n_eff: float

    # Explained variation
    r2: float
    adj_r2: float
    sigma: float
    rmse: float
    mae: float
    medape: float  # median absolute % error, price space

    # Complexity-penalised, on the effective sample size
    aic: float
    aicc: float
    bic: float
    d_aic: float = float("nan")
    d_aicc: float = float("nan")
    d_bic: float = float("nan")

    # Naive daily-n information criterion, kept to show why it is unusable
    aic_naive: float = float("nan")
    d_aic_naive: float = float("nan")

    # Residual diagnostics
    durbin_watson: float = float("nan")
    rho1: float = float("nan")
    skew: float = float("nan")
    excess_kurtosis: float = float("nan")

    # Band calibration
    coverage: float = float("nan")  # share of history inside the rainbow
    calibration_error: float = float("nan")  # mean |observed − nominal| occupancy


@dataclass
class Backtest:
    """Walk-forward out-of-sample forecast accuracy."""

    origins: int
    horizon_days: int
    rmse_log: float
    mae_log: float
    bias_log: float
    medape: float
    coverage: float
    skill: float  # vs a random-walk baseline
    failures: int = 0
    # Per-origin log errors and the origin indices they belong to, kept so the
    # paired bootstrap can align two models on the origins they share.
    errors: np.ndarray = field(default_factory=lambda: np.array([]))
    origin_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    # RMSE restricted to the origins every model forecast, so the ranking is
    # apples-to-apples even when some models failed on some origins. Equal to
    # rmse_log whenever nothing failed.
    rmse_common: float = float("nan")
    # Bootstrap interval for (this model's RMSE − the best model's RMSE).
    rmse_delta_low: float = float("nan")
    rmse_delta_high: float = float("nan")
    tied_with_best: bool = False


# --------------------------------------------------------------------------
# effective sample size
# --------------------------------------------------------------------------


def integrated_autocorrelation_time(residuals: np.ndarray) -> float:
    """Decorrelation time τ of a residual series, in observations.

    τ = 1 + 2·Σ ρ(m), summed while the autocorrelation stays above
    ``ACF_CUTOFF``, with a Bartlett taper on each term. Two residuals separated
    by more than roughly τ carry independent information.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 1.0

    centred = r - r.mean()
    acf = np.correlate(centred, centred, mode="full")[centred.size - 1 :]
    if acf[0] <= 0:
        return 1.0
    acf = acf / acf[0]

    tau = 1.0
    for lag in range(1, acf.size):
        if acf[lag] < ACF_CUTOFF:
            break
        tau += 2.0 * acf[lag] * (1.0 - lag / acf.size)
    return max(tau, 1.0)


def effective_sample_size(residuals: np.ndarray) -> float:
    """Number of independent observations the residual series is worth."""
    n = int(np.sum(np.isfinite(residuals)))
    return n / integrated_autocorrelation_time(residuals)


def reference_n_eff(inp) -> float:
    """The common effective sample size used for every model's AIC/BIC.

    Deliberately derived from one fixed reference fit — the power law — rather
    than from each model's own residuals. Per-model n_eff is circular and
    actively perverse: a model that fits badly leaves *more* autocorrelated
    residuals, which shrinks its own n_eff, which shrinks its likelihood
    penalty. Scoring that way ranks the two worst models here first.

    Anchoring to a fixed reference also keeps the numbers stable regardless of
    which subset of models the user selected.
    """
    log_t = np.log(inp.t_days_hist)
    design = np.column_stack([np.ones(inp.n_hist), log_t])
    coef, *_ = np.linalg.lstsq(design, inp.y, rcond=None)
    return effective_sample_size(inp.y - design @ coef)


# --------------------------------------------------------------------------
# in-sample metrics
# --------------------------------------------------------------------------


def _gaussian_log_likelihood(sigma_sq: float, n: float) -> float:
    """Gaussian log-likelihood at the MLE variance, for ``n`` observations."""
    return -0.5 * n * (np.log(2.0 * np.pi) + np.log(sigma_sq) + 1.0)


def evaluate(fit, inp, n_params: int, n_eff: float) -> Metrics:
    """Compute every in-sample statistic for one fitted model."""
    n = inp.n_hist
    price = np.exp(inp.y)
    fair = fit.price()[:n]
    residuals = inp.y - fit.center[:n]
    finite = residuals[np.isfinite(residuals)]

    sigma = float(np.std(finite))
    sigma_sq = float(np.mean(finite**2))
    ss_res = float(np.sum(finite**2))
    ss_tot = float(np.sum((inp.y - np.mean(inp.y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Textbook adjustment. Reported for completeness, but with n in the
    # thousands and k <= 6 it moves the fourth decimal at most - it does not
    # answer the "more parameters flatter the fit" objection.
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1)

    # +1 for the estimated residual variance.
    k_total = n_params + 1
    log_lik = _gaussian_log_likelihood(sigma_sq, n_eff)
    aic = 2.0 * k_total - 2.0 * log_lik
    denominator = n_eff - k_total - 1.0
    aicc = aic + (2.0 * k_total * (k_total + 1.0) / denominator if denominator > 0 else np.inf)
    bic = np.log(n_eff) * k_total - 2.0 * log_lik
    aic_naive = 2.0 * k_total - 2.0 * _gaussian_log_likelihood(sigma_sq, n)

    return Metrics(
        n=n,
        n_params=n_params,
        n_eff=n_eff,
        r2=r2,
        adj_r2=adj_r2,
        sigma=sigma,
        rmse=float(np.sqrt(sigma_sq)),
        mae=float(np.mean(np.abs(finite))),
        medape=float(np.median(np.abs(price / fair - 1.0)) * 100.0),
        aic=aic,
        aicc=aicc,
        bic=bic,
        aic_naive=aic_naive,
        durbin_watson=float(np.sum(np.diff(finite) ** 2) / np.sum(finite**2)),
        rho1=float(np.corrcoef(finite[:-1], finite[1:])[0, 1]),
        skew=float(stats.skew(finite)),
        excess_kurtosis=float(stats.kurtosis(finite)),
        coverage=_coverage(fit, price, n),
        calibration_error=_calibration_error(fit, price, n),
    )


def _coverage(fit, price: np.ndarray, n: int) -> float:
    """Share of observed prices falling inside the full rainbow."""
    lower = fit.band_prices()[0][:n]
    upper = fit.band_prices()[-1][:n]
    inside = (price >= lower) & (price <= upper)
    return float(np.mean(inside) * 100.0)


def _nominal_occupancy(fit) -> np.ndarray:
    """Share of observations each band should hold if the model is calibrated."""
    from models import BAND_SIGMA_STEP, NUM_BANDS, QUANTILE_EDGES

    if fit.band_mode == "quantile":
        # Quantile bands hold equal shares by construction.
        return np.diff(QUANTILE_EDGES)

    # Sigma bands assume log residuals are roughly normal.
    steps = (np.arange(NUM_BANDS + 1) - NUM_BANDS / 2.0) * BAND_SIGMA_STEP
    return np.diff(stats.norm.cdf(steps))


def _calibration_error(fit, price: np.ndarray, n: int) -> float:
    """Mean absolute gap between observed and nominal band occupancy, in points.

    The sigma bands assume log residuals are approximately normal. Where they
    are skewed, the rainbow is miscalibrated in a directly measurable way: this
    is how far off, averaged across the nine bands.
    """
    edges = [edge[:n] for edge in fit.band_prices()]
    observed = np.array(
        [np.mean((price >= edges[i]) & (price < edges[i + 1])) for i in range(len(edges) - 1)]
    )
    return float(np.mean(np.abs(observed - _nominal_occupancy(fit))) * 100.0)


def rank(metrics_by_key: dict) -> None:
    """Fill in ΔAIC / ΔAICc / ΔBIC, in place.

    Deliberately no Akaike weights. A weight is a probability that one model is
    best, and n_eff here is a heuristic - dressing it up as a probability would
    add exactly the false precision this module exists to strip out.
    """
    if not metrics_by_key:
        return

    for attribute, delta in (("aic", "d_aic"), ("aicc", "d_aicc"), ("bic", "d_bic")):
        values = [getattr(m, attribute) for m in metrics_by_key.values()]
        best = min(v for v in values if np.isfinite(v)) if any(np.isfinite(values)) else np.nan
        for m in metrics_by_key.values():
            setattr(m, delta, getattr(m, attribute) - best)

    naive = [m.aic_naive for m in metrics_by_key.values()]
    best_naive = min(naive)
    for m in metrics_by_key.values():
        m.d_aic_naive = m.aic_naive - best_naive


# --------------------------------------------------------------------------
# out-of-sample walk-forward backtest
# --------------------------------------------------------------------------


def backtest_origins(
    raw_data: pd.DataFrame,
    horizon_days: int = BACKTEST_HORIZON_DAYS,
    spacing_days: int = BACKTEST_SPACING_DAYS,
    start: pd.Timestamp = BACKTEST_START,
) -> list:
    """Forecast origins that leave a full horizon of data to score against."""
    last = raw_data["Date"].max()
    latest_origin = last - pd.Timedelta(days=horizon_days)
    origins = []
    cursor = start
    while cursor <= latest_origin:
        origins.append(cursor)
        cursor += pd.Timedelta(days=spacing_days)
    return origins


def walk_forward(
    specs,
    raw_data: pd.DataFrame,
    horizon_days: int = BACKTEST_HORIZON_DAYS,
    spacing_days: int = BACKTEST_SPACING_DAYS,
    start: pd.Timestamp = BACKTEST_START,
    progress=None,
) -> dict:
    """Refit every model on truncated history and score its forecasts.

    An **expanding** window is used, not a rolling one: every model here claims
    to describe the whole history since genesis, so discarding early data would
    misrepresent them.

    At each origin the models are refitted using only data up to that date, then
    asked for the price ``horizon_days`` later. Errors are measured in log space
    (where the models actually live) and summarised against a random-walk
    baseline, which for Bitcoin over a one-year horizon is a genuinely hard
    benchmark to beat.

    One known, small optimistic bias: ``btc_supply.HEIGHT_ANCHORS`` hard-codes
    all four observed halving dates, so a refit at a 2016 origin "knows" when the
    2020 and 2024 halvings landed. This touches only ``s2f`` and
    ``halving_cycle``, and only through the placement of future halvings - never
    through price. Halving dates are derivable from the block schedule years
    ahead, so a 2016 forecaster could have estimated them to within weeks; the
    leak is real but its magnitude is days of timing, not knowledge of the
    outcome.

    Returns:
        ``{model_key: Backtest}``.
    """
    from models import build_input, fit_model
    from plot import extend_dates

    origins = backtest_origins(raw_data, horizon_days, spacing_days, start)
    if not origins:
        return {}

    dates = raw_data["Date"]
    values = raw_data["Value"].to_numpy(dtype=float)
    months = int(np.ceil(horizon_days / 30.0)) + 2

    records = {spec.key: [] for spec in specs}
    failures = {spec.key: 0 for spec in specs}
    # Keyed by origin, so a model that failed on some origins is compared against
    # the baseline on exactly the origins it did produce - averaging the baseline
    # over all origins would flatter a model that fell over on the hard ones.
    baseline_errors = {}

    for origin_id, origin in enumerate(origins):
        position = origin_id + 1
        target = origin + pd.Timedelta(days=horizon_days)
        train_mask = dates <= origin
        if train_mask.sum() < 400:
            continue

        # Nearest actual observation to the target date.
        target_idx = int((dates - target).abs().to_numpy().argmin())
        actual = float(values[target_idx])
        if not np.isfinite(actual) or actual <= 0:
            continue

        history = raw_data[train_mask].reset_index(drop=True)
        extended = extend_dates(history, months=months)
        sub_input = build_input(history, extended)

        offsets = (pd.to_datetime(pd.Series(extended.to_numpy())) - target).abs()
        forecast_idx = int(offsets.to_numpy().argmin())

        # Random-walk baseline: the last observed price simply persists.
        baseline_errors[origin_id] = np.log(actual / float(history["Value"].iloc[-1]))

        if progress:
            progress(position, len(origins), origin)

        for spec in specs:
            try:
                fitted = fit_model(spec, sub_input)
                predicted = float(fitted.price()[forecast_idx])
                if not np.isfinite(predicted) or predicted <= 0:
                    raise ValueError("non-positive forecast")
                low = float(fitted.band_prices()[0][forecast_idx])
                high = float(fitted.band_prices()[-1][forecast_idx])
            except Exception:
                failures[spec.key] += 1
                continue
            records[spec.key].append(
                (
                    origin_id,
                    np.log(actual / predicted),
                    actual / predicted - 1.0,
                    low <= actual <= high,
                )
            )

    results = {}
    for spec in specs:
        rows = records[spec.key]
        if not rows:
            continue
        origin_ids = np.array([r[0] for r in rows])
        errors = np.array([r[1] for r in rows])
        relative = np.array([r[2] for r in rows])
        inside = np.array([r[3] for r in rows])
        mse = float(np.mean(errors**2))
        paired = np.array([baseline_errors[i] for i in origin_ids])
        baseline_mse = float(np.mean(paired**2)) if paired.size else np.nan
        results[spec.key] = Backtest(
            origins=len(rows),
            horizon_days=horizon_days,
            rmse_log=float(np.sqrt(mse)),
            mae_log=float(np.mean(np.abs(errors))),
            bias_log=float(np.mean(errors)),
            medape=float(np.median(np.abs(relative)) * 100.0),
            coverage=float(np.mean(inside) * 100.0),
            skill=float(1.0 - mse / baseline_mse) if baseline_mse > 0 else float("nan"),
            failures=failures[spec.key],
            errors=errors,
            origin_ids=origin_ids,
        )

    compare_out_of_sample(results, horizon_days, spacing_days)
    return results


def compare_out_of_sample(
    results: dict,
    horizon_days: int = BACKTEST_HORIZON_DAYS,
    spacing_days: int = BACKTEST_SPACING_DAYS,
    draws: int = 2000,
    seed: int = 0,
) -> None:
    """Attach a bootstrap interval for each model's RMSE gap to the best, in place.

    Forecast windows overlap — with a 365-day horizon and origins 182 days apart,
    consecutive forecasts share most of their evaluation period — so errors from
    neighbouring origins are not independent. A *moving-block* bootstrap
    resamples contiguous runs of origins instead of individual ones, which keeps
    that dependence intact. Blocks are sized to span one full horizon.

    Where the interval straddles zero, the model is statistically indistinguishable
    from the best one: with only a couple of dozen origins, most of this table's
    ordering is noise, and saying so is the point. The interval is deliberately
    conservative — see the Bonferroni note below — because the failure mode that
    matters here is claiming a difference that is not there.
    """
    scored = {k: b for k, b in results.items() if b.errors.size > 1}
    if len(scored) < 2:
        return

    # Rank on origins EVERY model forecast, not on each model's own subset. A
    # model that fails on the hard origins would otherwise post the lowest RMSE
    # on an easier sample and be crowned the reference everyone is measured
    # against.
    common = scored[next(iter(scored))].origin_ids
    for backtest in scored.values():
        common = np.intersect1d(common, backtest.origin_ids)
    if common.size < 3:
        return

    def squared_on_common(backtest):
        return backtest.errors[np.isin(backtest.origin_ids, common)] ** 2

    for backtest in scored.values():
        backtest.rmse_common = float(np.sqrt(np.mean(squared_on_common(backtest))))

    best_key = min(scored, key=lambda k: scored[k].rmse_common)
    best = scored[best_key]
    theirs = squared_on_common(best)

    n = common.size
    # Blocks must be shorter than the sample, or every "resample" is the original
    # sample and the interval collapses to zero width - which would declare every
    # other model significantly worse with total confidence. Reachable from a
    # long --horizon-days, where the origin count falls below the block length.
    block = max(1, min(int(np.ceil(horizon_days / max(spacing_days, 1))), n - 1))

    # Each model is compared against the model that won on this same sample, so
    # the comparisons are neither independent nor pre-designated. Left uncorrected
    # the procedure suffers the winner's curse and flags far more models as
    # "clearly worse" than it should. A Bonferroni split across the comparisons
    # keeps the family-wise claim honest; it errs toward calling things tied,
    # which is the right direction for a table this small.
    comparisons = max(len(scored) - 1, 1)
    alpha = 0.10 / comparisons
    lower_pct, upper_pct = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)

    for candidate in scored.values():
        mine = squared_on_common(candidate)
        starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(draws, -1)[:, :n]

        deltas = np.sqrt(mine[idx].mean(axis=1)) - np.sqrt(theirs[idx].mean(axis=1))
        low, high = np.percentile(deltas, [lower_pct, upper_pct])
        candidate.rmse_delta_low = float(low)
        candidate.rmse_delta_high = float(high)
        candidate.tied_with_best = bool(low <= 0.0 <= high)

    best.tied_with_best = True


def parameter_premise(n_params: list, r2: list) -> dict:
    """Check how strongly extra parameters actually buy in-sample fit here.

    "More parameters means better R²" is true as a tendency but not a law, and
    it is worth measuring rather than assuming: it only holds strictly for
    *nested* models, and these thirteen are mostly not nested.
    """
    from scipy.stats import spearmanr

    k = np.asarray(n_params, dtype=float)
    r = np.asarray(r2, dtype=float)
    pairs = [(i, j) for i in range(len(k)) for j in range(len(k)) if k[i] > k[j]]
    worse = sum(1 for i, j in pairs if r[i] < r[j])
    rho, p_value = spearmanr(k, r) if len(k) > 2 else (np.nan, np.nan)
    return {
        "spearman": float(rho),
        "p_value": float(p_value),
        "pairs": len(pairs),
        "inversions": worse,
    }
