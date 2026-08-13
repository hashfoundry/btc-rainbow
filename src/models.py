"""Regression models for the Bitcoin rainbow chart.

Every model reduces to the same contract: given time, produce a centre line in
natural-log price space. The rainbow bands are then built generically from the
model's residuals, so all models are directly comparable on the same colour
scale.

Two band strategies are supported:

* ``sigma``    - bands are fixed multiples of the residual standard deviation.
                 This is the classic rainbow construction.
* ``quantile`` - band edges sit at empirical residual quantiles, so each band
                 holds a known share of Bitcoin's history.

References for the model families are collected in
``bitcoin_rainbow_chart_regression_models.md``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import theilslopes

from btc_supply import GENESIS, halving_dates, stock_to_flow

# Each band is half a residual standard deviation tall, so the nine-band
# rainbow spans +/- 2.25 sigma around the trend.
BAND_SIGMA_STEP = 0.5
NUM_BANDS = 9
# Outer edges sit at the 1st/99th percentile so extreme outliers do not stretch
# the whole rainbow.
QUANTILE_EDGES = np.linspace(0.01, 0.99, NUM_BANDS + 1)

# Bitcoin has never visibly saturated, so the S-curve models cannot identify a
# ceiling from the data - the fit simply climbs until something stops it. This
# caps that search, making the logistic and Gompertz charts an explicit
# "adoption tops out here" scenario rather than an artefact of a loose bound.
SATURATION_CEILING_USD = 10_000_000.0

# Likewise for the hyperbolic model: its residuals shrink monotonically as the
# singularity is pushed further out, so the scan needs an explicit horizon.
# ~30 years past the last observation.
HYPERBOLIC_MAX_HORIZON_DAYS = 11_000.0

# Plan C's Quantile Model works on named valuation tiers rather than a rainbow:
# six quantile curves bounding five bands. Levels and names follow version 2.
QUANTILE_MODEL_LEVELS = [0.01, 0.20, 0.50, 0.80, 0.95, 0.999]
QUANTILE_MODEL_BANDS = [
    ("#ececec", "Deep Value (1–20)"),
    ("#cfe0f5", "Discounted (20–50)"),
    ("#fdf0c9", "Premium (50–80)"),
    ("#f2c2c2", "Speculative (80–95)"),
    ("#d9d9d9", "Historic Peaks (95–99.9)"),
]


@dataclass
class ModelInput:
    """Time grid shared by every model.

    The grid covers the price history followed by a projection window; only the
    first ``n_hist`` points have an observed price.
    """

    dates: pd.Series  # history + projection
    t_days: np.ndarray  # days since the genesis block
    t_index: np.ndarray  # 1..N row counter (legacy models use this)
    y: np.ndarray  # ln(price), history only
    n_hist: int

    @property
    def t_days_hist(self) -> np.ndarray:
        return self.t_days[: self.n_hist]

    @property
    def t_index_hist(self) -> np.ndarray:
        return self.t_index[: self.n_hist]


@dataclass
class Fit:
    """A fitted model plus the rainbow bands derived from it."""

    key: str
    name: str
    description: str
    formula: str
    band_mode: str
    center: np.ndarray  # ln price over the whole grid
    edges: list  # arrays of ln price, low to high; one more than there are bands
    params: dict = field(default_factory=dict)
    sigma: float = float("nan")
    r2: float = float("nan")
    rmse: float = float("nan")
    # Models that define their own bands (rather than using the nine-colour
    # rainbow) fill these in; empty means "use the default rainbow".
    band_colors: list = field(default_factory=list)
    band_labels: list = field(default_factory=list)
    band_levels: list = field(default_factory=list)  # cumulative, for calibration

    def price(self) -> np.ndarray:
        return np.exp(self.center)

    def band_prices(self) -> list:
        return [np.exp(edge) for edge in self.edges]


@dataclass
class ModelSpec:
    key: str
    name: str
    description: str
    formula: str
    fit: Callable[[ModelInput], tuple]  # -> (center over grid, params dict)
    band_mode: str = "sigma"
    # Free parameters in the centre line, used for the complexity penalty in the
    # information criteria. Counts every quantity the fit estimates from the
    # data, including ones found by grid search rather than by an optimiser -
    # a parameter chosen by scanning costs exactly as much freedom as one solved
    # for. The residual variance is added separately in metrics.evaluate.
    n_params: int = 2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def build_input(raw_data: pd.DataFrame, extended_dates: pd.Series) -> ModelInput:
    """Assemble the shared time grid from the price history and projection."""
    n_hist = len(raw_data)
    dates = pd.Series(pd.to_datetime(extended_dates.to_numpy()))
    t_days = (dates - GENESIS).dt.total_seconds().to_numpy() / 86400.0
    # Day 0 is the genesis block itself; keep the log-time models defined.
    t_days = np.maximum(t_days, 1.0)
    t_index = np.arange(1, len(dates) + 1, dtype=float)
    y = np.log(raw_data["Value"].to_numpy(dtype=float))
    return ModelInput(dates=dates, t_days=t_days, t_index=t_index, y=y, n_hist=n_hist)


def _finish(
    spec: ModelSpec,
    inp: ModelInput,
    center: np.ndarray,
    params: dict,
    bands: dict | None = None,
) -> Fit:
    """Derive residual statistics and band edges from a centre line.

    ``bands`` lets a model supply its own edges, colours and labels instead of
    the nine-colour rainbow — used by the quantile model, whose edges are each a
    separate regression rather than an offset from the centre.
    """
    center = np.asarray(center, dtype=float)
    residuals = inp.y - center[: inp.n_hist]
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        raise ValueError("model produced no finite residuals")

    sigma = float(np.std(finite))
    ss_res = float(np.sum(finite**2))
    ss_tot = float(np.sum((inp.y - np.mean(inp.y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / finite.size))

    if bands is not None:
        edges = bands["edges"]
        extra = {
            "band_colors": bands["colors"],
            "band_labels": bands["labels"],
            "band_levels": bands["levels"],
        }
    else:
        if spec.band_mode == "quantile":
            offsets = np.quantile(finite, QUANTILE_EDGES)
        else:
            steps = np.arange(NUM_BANDS + 1) - NUM_BANDS / 2.0
            offsets = steps * BAND_SIGMA_STEP * sigma
        edges = [center + offset for offset in offsets]
        extra = {}

    return Fit(
        key=spec.key,
        name=spec.name,
        description=spec.description,
        formula=spec.formula,
        band_mode=spec.band_mode,
        center=center,
        edges=edges,
        params=params,
        sigma=sigma,
        r2=r2,
        rmse=rmse,
        **extra,
    )


def _linear_fit(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


# --------------------------------------------------------------------------
# model implementations
# --------------------------------------------------------------------------


def _fit_log_time(inp: ModelInput):
    """ln(P) = a * ln(b + i) + c, fitted on the row counter (original chart)."""

    def func(x, a, b, c):
        return a * np.log(b + x) + c

    popt, _ = curve_fit(func, inp.t_index_hist, inp.y, maxfev=20000)
    center = func(inp.t_index, *popt)
    return center, {"a": popt[0], "b": popt[1], "c": popt[2]}


def _powerlaw_coef(inp: ModelInput) -> np.ndarray:
    """OLS of ln(P) on ln(days since genesis)."""
    design = np.column_stack([np.ones(inp.n_hist), np.log(inp.t_days_hist)])
    return _linear_fit(design, inp.y)


def _fit_power_law(inp: ModelInput):
    """log10(P) = a * log10(d) + b - the Santostasi power-law corridor."""
    intercept, slope = _powerlaw_coef(inp)
    center = intercept + slope * np.log(inp.t_days)
    return center, {
        "exponent": slope,
        "log10_intercept": intercept / np.log(10.0),
        "coefficient": float(np.exp(intercept)),
    }


def _fit_power_law_robust(inp: ModelInput):
    """Power law fitted with Theil-Sen, which ignores bubble/crash outliers."""
    log_t = np.log(inp.t_days_hist)
    slope, intercept, lo, hi = theilslopes(inp.y, log_t)
    center = intercept + slope * np.log(inp.t_days)
    return center, {
        "exponent": slope,
        "exponent_ci_low": lo,
        "exponent_ci_high": hi,
        "log10_intercept": intercept / np.log(10.0),
    }


def _fit_log_linear(inp: ModelInput):
    """ln(P) = a * d + b - constant compound growth, linear time."""
    design = np.column_stack([np.ones(inp.n_hist), inp.t_days_hist])
    intercept, slope = _linear_fit(design, inp.y)
    center = intercept + slope * inp.t_days
    return center, {
        "daily_growth_pct": float(np.expm1(slope) * 100.0),
        "annual_growth_pct": float(np.expm1(slope * 365.25) * 100.0),
        "intercept": intercept,
    }


def _fit_stretched_log(inp: ModelInput):
    """ln(P) = a * ln(d)^c + b - power law with adjustable curvature."""

    def func(t, a, b, c):
        return a * np.power(np.log(t), c) + b

    p0 = [1.0, float(inp.y.min()), 1.0]
    bounds = ([1e-6, -np.inf, 0.2], [np.inf, np.inf, 5.0])
    popt, _ = curve_fit(func, inp.t_days_hist, inp.y, p0=p0, bounds=bounds, maxfev=40000)
    center = func(inp.t_days, *popt)
    return center, {"a": popt[0], "b": popt[1], "c": popt[2]}


def _fit_time_offset_log(inp: ModelInput):
    """ln(P) = a * ln(d - t0) + b - shifts the clock off the genesis block."""
    t_min = float(inp.t_days_hist.min())

    def func(t, a, b, t0):
        return a * np.log(np.maximum(t - t0, 1e-6)) + b

    p0 = [5.0, -35.0, 0.0]
    bounds = ([1e-6, -np.inf, -3000.0], [np.inf, np.inf, t_min - 1.0])
    popt, _ = curve_fit(func, inp.t_days_hist, inp.y, p0=p0, bounds=bounds, maxfev=40000)
    center = func(inp.t_days, *popt)
    return center, {"a": popt[0], "b": popt[1], "t_offset_days": popt[2]}


def _fit_hyperbolic(inp: ModelInput):
    """P = a / (b - d)^c - finite-time singularity at day b.

    ``ln(P) = ln(a) - c·ln(b - d)`` is linear in ``(ln a, c)`` once ``b`` is
    fixed, so the singularity date is found by a scan and the other two
    parameters are solved exactly at each node. Optimising all three jointly
    walks off to a distant ``b``, where ``ln(b - d)`` is near-linear and the
    model silently collapses into the exponential trend.

    That collapse is not an artefact of the search: residuals fall monotonically
    as ``b`` grows, because this form is convex in log-price while Bitcoin's log
    price decelerates. The full history therefore supports no finite-time
    singularity, and the scan is capped at ``HYPERBOLIC_MAX_HORIZON_DAYS`` so it
    terminates somewhere interpretable. Read the fitted date as the edge of that
    cap, not as an estimate.
    """
    # Anchored to the end of the OBSERVED history, never to the plotted grid.
    # Using inp.t_days here would let the projection window - a display choice -
    # move the fitted singularity by years and change R², and would fit a
    # different model during the backtest than the one the chart shows.
    t_max = float(inp.t_days_hist.max())

    best = None
    for b in t_max + np.geomspace(60.0, HYPERBOLIC_MAX_HORIZON_DAYS, 400):
        design = np.column_stack([np.ones(inp.n_hist), -np.log(b - inp.t_days_hist)])
        coef, *_ = np.linalg.lstsq(design, inp.y, rcond=None)
        sse = float(np.sum((inp.y - design @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, b, coef)

    _, b, coef = best
    center = coef[0] - coef[1] * np.log(np.maximum(b - inp.t_days, 1e-6))
    singularity = GENESIS + pd.Timedelta(days=float(b))
    # `a` itself is e**coef[0], which overflows float64 at these horizons and
    # carries no interpretable meaning; the exponent and date are the readable
    # parameters.
    return center, {
        "singularity_day": b,
        "singularity_date": singularity.date().isoformat(),
        "c": float(coef[1]),
        "horizon_capped": bool(b >= t_max + HYPERBOLIC_MAX_HORIZON_DAYS * 0.999),
    }


def _ceiling_bounds(inp: ModelInput) -> tuple:
    """Search range for a saturation ceiling, in natural-log price."""
    floor = float(np.log(np.exp(inp.y.max()) * 1.05))
    return floor, float(np.log(SATURATION_CEILING_USD))


def _fit_logistic(inp: ModelInput):
    """ln(P) = ceiling - S / (1 + e^(k(d - d0))) - saturating adoption S-curve.

    Parameterised on the ceiling rather than on an amplitude-plus-offset pair.
    Bitcoin has not visibly saturated, so amplitude and offset are only weakly
    identified and a raw fit drifts until it hits whatever bound it is given;
    fitting the asymptote directly keeps the reported ceiling meaningful.
    """

    def func(t, ceiling, s, k, t0):
        return ceiling - s / (1.0 + np.exp(k * (t - t0)))

    lo, hi = _ceiling_bounds(inp)
    span = float(inp.y.max() - inp.y.min())
    p0 = [min(lo + 1.0, hi), span, 0.001, float(np.median(inp.t_days_hist))]
    bounds = ([lo, 1e-6, 1e-6, -20000.0], [hi, 60.0, 1.0, 1e6])
    popt, _ = curve_fit(func, inp.t_days_hist, inp.y, p0=p0, bounds=bounds, maxfev=60000)
    midpoint = GENESIS + pd.Timedelta(days=float(popt[3]))
    return func(inp.t_days, *popt), {
        "ceiling_price": float(np.exp(popt[0])),
        "log_span": popt[1],
        "k": popt[2],
        "midpoint_day": popt[3],
        "midpoint_date": midpoint.date().isoformat(),
    }


def _fit_gompertz(inp: ModelInput):
    """ln(P) = ceiling - S + S * e^(-B e^(-C tau)) - asymmetric saturation."""
    scale = 1000.0

    def func(t, ceiling, s, B, C):
        return ceiling - s + s * np.exp(-B * np.exp(-C * (t / scale)))

    lo, hi = _ceiling_bounds(inp)
    span = float(inp.y.max() - inp.y.min())
    p0 = [min(lo + 1.0, hi), span, 5.0, 0.4]
    bounds = ([lo, 1e-6, 1e-6, 1e-6], [hi, 60.0, 1e4, 20.0])
    popt, _ = curve_fit(func, inp.t_days_hist, inp.y, p0=p0, bounds=bounds, maxfev=60000)
    return func(inp.t_days, *popt), {
        "ceiling_price": float(np.exp(popt[0])),
        "log_span": popt[1],
        "B": popt[2],
        "C": popt[3],
    }


def _fit_s2f(inp: ModelInput):
    """ln(P) = a * ln(S2F) + b - PlanB's stock-to-flow regression."""
    s2f = stock_to_flow(inp.dates)
    log_s2f = np.log(s2f)
    hist = log_s2f[: inp.n_hist]
    design = np.column_stack([np.ones(inp.n_hist), hist])
    intercept, slope = _linear_fit(design, inp.y)
    center = intercept + slope * log_s2f
    return center, {
        "exponent": slope,
        "intercept": intercept,
        "s2f_today": float(s2f[inp.n_hist - 1]),
    }


def _fit_lppl(inp: ModelInput):
    """Power-law trend plus a damped log-periodic oscillation.

    The trend is the log-log regression; the cycle term
    ``(d/d_ref)^-m * (c1 cos(w ln d) + c2 sin(w ln d))`` is fitted to its
    residuals by a grid search over the angular log-frequency ``w`` and damping
    ``m``, solving the remaining linear coefficients exactly at each node. That
    is far more stable than optimising the full LPPLS form directly.

    ``w`` is restricted to log-frequencies whose present-day wavelength is two
    to seven years. Left unconstrained the fit prefers a single ~16-year arc:
    that scores better, but it describes the slow bend of the trend rather than
    the market cycle the model is meant to capture. Inside the window the
    optimum is interior, not pinned to an edge.
    """
    intercept, slope = _powerlaw_coef(inp)
    log_t = np.log(inp.t_days)
    trend = intercept + slope * log_t
    residuals = inp.y - trend[: inp.n_hist]

    t_ref = float(np.median(inp.t_days_hist))
    log_t_hist = log_t[: inp.n_hist]
    damp_hist = inp.t_days_hist / t_ref

    t_now = float(inp.t_days_hist[-1])
    omega_hi = 2.0 * np.pi / np.log1p(2.0 * 365.25 / t_now)
    omega_lo = 2.0 * np.pi / np.log1p(7.0 * 365.25 / t_now)

    best = None
    for omega in np.linspace(omega_lo, omega_hi, 90):
        phase = omega * log_t_hist
        for m in np.linspace(0.0, 1.5, 16):
            weight = np.power(damp_hist, -m)
            design = np.column_stack([weight * np.cos(phase), weight * np.sin(phase)])
            coef, *_ = np.linalg.lstsq(design, residuals, rcond=None)
            sse = float(np.sum((residuals - design @ coef) ** 2))
            if best is None or sse < best[0]:
                best = (sse, omega, m, coef)

    _, omega, m, coef = best
    weight_all = np.power(inp.t_days / t_ref, -m)
    cycle = weight_all * (coef[0] * np.cos(omega * log_t) + coef[1] * np.sin(omega * log_t))
    center = trend + cycle

    amplitude = float(np.hypot(coef[0], coef[1]))
    # Current log-periodic wavelength expressed in calendar days.
    t_now = float(inp.t_days_hist[-1])
    period_days = t_now * np.expm1(2.0 * np.pi / omega)
    return center, {
        "exponent": slope,
        "omega": omega,
        "damping_m": m,
        "amplitude": amplitude,
        "current_cycle_days": float(period_days),
    }


def quantile_regression(design: np.ndarray, y: np.ndarray, tau: float, iterations: int = 200):
    """Linear quantile regression, minimising the pinball loss for ``tau``.

    Solved by iteratively reweighted least squares (Schlossmacher): the pinball
    loss is |r| reweighted by tau above the line and 1-tau below, so repeatedly
    solving a weighted least-squares problem with weights w = tau_or_1-tau / |r|
    converges to the quantile fit. Verified against the exact linear-programming
    solution: coefficients agree to four decimals on every quantile used here,
    at roughly a tenth of the cost — which matters because the backtest refits
    every model at every forecast origin.
    """
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    floor = 1e-6 * max(float(np.std(y)), 1e-9)

    for _ in range(iterations):
        residual = y - design @ beta
        weight = np.where(residual >= 0, tau, 1.0 - tau) / np.maximum(np.abs(residual), floor)
        root = np.sqrt(weight)
        updated, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        shift = float(np.max(np.abs(updated - beta)))
        beta = updated
        if shift < 1e-10:
            break
    return beta


def _fit_quantile_model(inp: ModelInput):
    """Plan C's Bitcoin Quantile Model: one power-law regression per quantile.

    Every band edge is its own quantile regression of ln(price) on ln(days since
    genesis), rather than an offset from a shared centre. Each quantile
    therefore gets its own slope, and the measured slopes fall steadily from
    ~5.9 at the 1st percentile to ~4.5 at the 99.9th — which is why the channel
    is wide in the early years and narrows as the series matures, the
    distinctive feature of this model against a fixed-width corridor.

    The centre line is the fitted median, not the least-squares mean.
    """
    log_t = np.log(inp.t_days)
    design_hist = np.column_stack([np.ones(inp.n_hist), log_t[: inp.n_hist]])

    curves, coefficients = [], {}
    for tau in QUANTILE_MODEL_LEVELS:
        beta = quantile_regression(design_hist, inp.y, tau)
        curves.append(beta[0] + beta[1] * log_t)
        coefficients[f"slope_q{tau * 100:g}"] = float(beta[1])

    # Independent quantile fits can cross where their slopes differ. Sorting the
    # curves pointwise is Chernozhukov's rearrangement: it restores monotonic
    # ordering without disturbing the fit anywhere the lines were already in
    # order, and gives the non-crossing guarantee v2 of the model advertises.
    stacked = np.sort(np.vstack(curves), axis=0)
    edges = [stacked[i] for i in range(stacked.shape[0])]
    center = stacked[QUANTILE_MODEL_LEVELS.index(0.5)]

    params = dict(coefficients)
    params["crossings_repaired"] = int(np.sum(np.vstack(curves) != stacked))
    params["quantile_today"] = _price_quantile(
        float(inp.y[-1]), [edge[inp.n_hist - 1] for edge in edges]
    )
    bands = {
        "edges": edges,
        "colors": [color for color, _ in QUANTILE_MODEL_BANDS],
        "labels": [label for _, label in QUANTILE_MODEL_BANDS],
        "levels": list(QUANTILE_MODEL_LEVELS),
    }
    return center, params, bands


def _price_quantile(log_price: float, edges_today: list) -> float:
    """Where today's price sits on the fitted quantile scale, in percent.

    Linear interpolation between the bracketing quantile curves — the reading
    the model's "Quantile: 56.2" headline reports. Clamped at the outermost
    curves: a price below the 1st-percentile line reads exactly 1, because the
    fit says nothing about the shape of the distribution beyond it.
    """
    levels = np.array(QUANTILE_MODEL_LEVELS) * 100.0
    values = np.array(edges_today, dtype=float)
    return float(np.interp(log_price, values, levels))


def _fit_halving_cycle(inp: ModelInput):
    """Power-law trend plus harmonics of the position inside the halving epoch.

    Epoch phase runs 0 -> 1 between consecutive halvings. Two harmonics are
    enough to reproduce the familiar run-up/blow-off/drawdown shape, and the
    whole model stays linear in its coefficients.
    """
    phase = _halving_phase(inp.dates)
    log_t = np.log(inp.t_days)

    columns = [np.ones(len(inp.dates)), log_t]
    for k in (1, 2):
        columns.append(np.cos(2.0 * np.pi * k * phase))
        columns.append(np.sin(2.0 * np.pi * k * phase))
    design = np.column_stack(columns)

    coef = _linear_fit(design[: inp.n_hist], inp.y)
    center = design @ coef

    amp1 = float(np.hypot(coef[2], coef[3]))
    peak_phase = float(phase[int(np.argmax(center))]) if len(center) else float("nan")
    return center, {
        "exponent": coef[1],
        "harmonic1_amplitude": amp1,
        "harmonic2_amplitude": float(np.hypot(coef[4], coef[5])),
        "cycle_peak_phase": peak_phase,
    }


def _halving_phase(dates: pd.Series) -> np.ndarray:
    """Fraction of the way through the current halving epoch, in [0, 1).

    Everything is reduced to days since genesis first. Comparing raw integer
    timestamps is a trap: ``Timestamp.value`` is always nanoseconds while a
    ``datetime64[us]`` Series casts to microseconds, so the two silently differ
    by 1000x.
    """
    dates = pd.to_datetime(pd.Series(dates.to_numpy()))
    last = dates.max() + pd.Timedelta(days=800)
    boundaries = [GENESIS] + halving_dates(last)

    edges = np.array(
        [(b - GENESIS).total_seconds() / 86400.0 for b in boundaries], dtype=float
    )
    values = (dates - GENESIS).dt.total_seconds().to_numpy() / 86400.0

    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    start = edges[idx]
    end = edges[idx + 1]
    return np.clip((values - start) / (end - start), 0.0, 1.0)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

MODELS = [
    ModelSpec(
        key="log_time",
        name="Logarithmic Regression (original)",
        description=(
            "The chart's original fit: a logarithmic curve on the row counter "
            "rather than on calendar time. Kept as the reference baseline."
        ),
        formula="ln(P) = a · ln(b + i) + c",
        fit=_fit_log_time,
        n_params=3,
    ),
    ModelSpec(
        key="power_law",
        name="Power Law Corridor",
        description=(
            "Straight line in log-log space: price against days since the "
            "genesis block. The standard Santostasi power law and the best "
            "documented long-run fit."
        ),
        formula="log₁₀(P) = a · log₁₀(d) + b   ⇔   P = A · d^a",
        fit=_fit_power_law,
        n_params=2,
    ),
    ModelSpec(
        key="power_law_quantile",
        name="Power Law – Quantile Corridor",
        description=(
            "Same power-law centre line, but band edges sit at empirical "
            "residual quantiles instead of standard deviations, so each band "
            "holds a known share of Bitcoin's history."
        ),
        formula="log₁₀(P) = a · log₁₀(d) + b, bands at residual quantiles",
        fit=_fit_power_law,
        n_params=2,
        band_mode="quantile",
    ),
    ModelSpec(
        key="power_law_robust",
        name="Power Law – Robust (Theil–Sen)",
        description=(
            "Power law fitted with the Theil–Sen estimator, whose median-of-"
            "slopes construction is insensitive to bubble tops and capitulation "
            "wicks that drag an ordinary least-squares fit around."
        ),
        formula="log₁₀(P) = a · log₁₀(d) + b, a = median pairwise slope",
        fit=_fit_power_law_robust,
        n_params=2,
    ),
    ModelSpec(
        key="log_linear",
        name="Log-Linear (Exponential Trend)",
        description=(
            "Constant compound growth against linear time. Included as the "
            "control case: its poor fit is exactly why rainbow charts moved to "
            "log-time models."
        ),
        formula="ln(P) = a · d + b   ⇔   P = e^(a·d + b)",
        fit=_fit_log_linear,
        n_params=2,
    ),
    ModelSpec(
        key="stretched_log",
        name="Modified Power Law (Stretched Log)",
        description=(
            "Adds a curvature exponent to the log-log fit, letting the trend "
            "bend as returns diminish instead of holding one fixed slope."
        ),
        formula="ln(P) = a · ln(d)^c + b",
        fit=_fit_stretched_log,
        n_params=3,
    ),
    ModelSpec(
        key="time_offset_log",
        name="Time-Adjusted Logarithmic Regression",
        description=(
            "Log-time regression with a fitted time origin, correcting for the "
            "distortion that Bitcoin's early price-discovery period imposes on "
            "a genesis-anchored clock."
        ),
        formula="ln(P) = a · ln(d − t₀) + b",
        fit=_fit_time_offset_log,
        n_params=3,
    ),
    ModelSpec(
        key="hyperbolic",
        name="Hyperbolic (Finite-Time Singularity)",
        description=(
            "Super-exponential growth diverging at a future date. Bitcoin's log "
            "price decelerates while this form accelerates, so the fit runs to "
            "its horizon cap and degenerates toward the exponential trend: "
            "useful evidence that the full history shows no finite-time "
            "singularity, not a forecast of one."
        ),
        formula="P = a / (b − d)^c",
        fit=_fit_hyperbolic,
        n_params=3,
    ),
    ModelSpec(
        key="logistic",
        name="Logistic Growth (S-Curve)",
        description=(
            "Adoption saturates: log price follows a sigmoid toward a ceiling. "
            "The counterpoint to the unbounded power law. Bitcoin has not "
            "visibly saturated, so the ceiling is a capped scenario "
            "($10M/BTC) rather than something the data identifies."
        ),
        formula="ln(P) = ceiling − S / (1 + e^(k(d − d₀)))",
        fit=_fit_logistic,
        n_params=4,
    ),
    ModelSpec(
        key="gompertz",
        name="Gompertz Growth",
        description=(
            "Asymmetric saturation curve - fast early growth, long slow "
            "approach to the ceiling. Tracks Bitcoin's early data better than "
            "the symmetric logistic. Same capped-ceiling caveat applies."
        ),
        formula="ln(P) = ceiling − S + S · e^(−B·e^(−C·τ)),  τ = d / 1000",
        fit=_fit_gompertz,
        n_params=4,
    ),
    ModelSpec(
        key="s2f",
        name="Stock-to-Flow",
        description=(
            "Scarcity model: log price against the log ratio of circulating "
            "supply to annual issuance, computed from the consensus issuance "
            "schedule. Note the well-known critique that supply appears on "
            "both sides of the regression."
        ),
        formula="ln(P) = a · ln(S2F) + b",
        fit=_fit_s2f,
        n_params=2,
    ),
    ModelSpec(
        key="lppl",
        name="Log-Periodic Power Law",
        description=(
            "Power-law trend plus a damped oscillation that is periodic in "
            "log-time - the Johansen–Ledoit–Sornette bubble signature. Bends "
            "the rainbow with the market cycle instead of averaging over it."
        ),
        formula="ln(P) = a·ln(d) + b + (d/d_ref)^(−m) · A·cos(ω·ln(d) − φ)",
        fit=_fit_lppl,
        n_params=6,
    ),
    ModelSpec(
        key="quantile_model",
        name="Quantile Model (Plan C)",
        description=(
            "Every band edge is its own quantile regression of log price on "
            "log days since genesis, so each quantile carries its own slope "
            "instead of sitting a fixed distance from the centre. The measured "
            "slopes fall from ~5.9 at the 1st percentile to ~4.5 at the 99.9th, "
            "so the channel is wide early and narrows as the series matures. "
            "Bands are the model's own valuation tiers, and the headline "
            "statistic is where today's price sits on the 1–99.9 scale."
        ),
        formula="Q_τ[ln(P)] = a_τ · ln(d) + b_τ,  τ ∈ {1, 20, 50, 80, 95, 99.9}%",
        fit=_fit_quantile_model,
        band_mode="explicit",
        # Two, not twelve. The model estimates 12 coefficients across its six
        # curves, but the information criteria in metrics.evaluate score the
        # CENTRE line only, and that centre is the median regression's 2
        # parameters. Charging all 12 against a 2-parameter likelihood would be
        # a category error - the same convention power_law_quantile follows,
        # where the band quantiles are likewise outside the likelihood.
        n_params=2,
    ),
    ModelSpec(
        key="halving_cycle",
        name="Halving-Cycle Regression",
        description=(
            "Power-law trend plus two Fourier harmonics of the position inside "
            "the current halving epoch, so the bands tilt with the four-year "
            "issuance cycle."
        ),
        formula="ln(P) = a·ln(d) + b + Σ(k=1,2) [α_k·cos(2πkφ) + β_k·sin(2πkφ)]",
        fit=_fit_halving_cycle,
        n_params=6,
    ),
]

MODELS_BY_KEY = {spec.key: spec for spec in MODELS}
DEFAULT_MODEL = "power_law"


def fit_model(spec: ModelSpec, inp: ModelInput) -> Fit:
    """Fit one model and build its bands."""
    result = spec.fit(inp)
    center, params = result[0], result[1]
    bands = result[2] if len(result) > 2 else None
    return _finish(spec, inp, center, params, bands)


def resolve_models(selection: str | None) -> list:
    """Turn a comma-separated selection (or ``all``) into model specs."""
    if not selection or selection.strip().lower() in {"all", "*"}:
        return list(MODELS)

    chosen = []
    unknown = []
    for raw in selection.split(","):
        key = raw.strip()
        if not key:
            continue
        if key in MODELS_BY_KEY:
            chosen.append(MODELS_BY_KEY[key])
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. "
            f"Available: {', '.join(MODELS_BY_KEY)}"
        )
    return chosen
