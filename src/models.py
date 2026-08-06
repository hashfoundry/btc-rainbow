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
    edges: list  # NUM_BANDS + 1 arrays of ln price, low to high
    params: dict = field(default_factory=dict)
    sigma: float = float("nan")
    r2: float = float("nan")
    rmse: float = float("nan")

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


def _finish(spec: ModelSpec, inp: ModelInput, center: np.ndarray, params: dict) -> Fit:
    """Derive residual statistics and rainbow band edges from a centre line."""
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

    if spec.band_mode == "quantile":
        offsets = np.quantile(finite, QUANTILE_EDGES)
    else:
        steps = np.arange(NUM_BANDS + 1) - NUM_BANDS / 2.0
        offsets = steps * BAND_SIGMA_STEP * sigma

    edges = [center + offset for offset in offsets]
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
    t_max = float(inp.t_days.max())

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
    ),
]

MODELS_BY_KEY = {spec.key: spec for spec in MODELS}
DEFAULT_MODEL = "power_law"


def fit_model(spec: ModelSpec, inp: ModelInput) -> Fit:
    """Fit one model and build its rainbow bands."""
    center, params = spec.fit(inp)
    return _finish(spec, inp, center, params)


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
