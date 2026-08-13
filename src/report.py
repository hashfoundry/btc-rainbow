"""HTML output: the shared stylesheet, the comparison tables and the page shells.

Both the index and every individual chart page carry the same three statistics
tables covering all models, so whichever chart you open you can see where that
model sits against the rest.
"""

import html

CSS = """
  :root {
    --surface: #ffffff; --raised: #f7f7f8; --ink: #16181d; --muted: #6b7280;
    --line: #e5e7eb; --accent: #b45309; --mark: #fdf6ec;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface: #101216; --raised: #181b21; --ink: #eceef2; --muted: #9aa1ad;
      --line: #2a2e37; --accent: #f0b429; --mark: #241f14;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 40px 24px 80px; background: var(--surface); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  main { max-width: 1180px; margin: 0 auto; }
  h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: -0.02em; }
  h2 { font-size: 18px; margin: 34px 0 4px; letter-spacing: -0.01em; }
  h2 .hint { font-weight: 400; color: var(--muted); font-size: 13px; letter-spacing: 0; }
  p.sub { color: var(--muted); margin: 0 0 28px; }
  .stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 32px; }
  .stat {
    flex: 1 1 160px; background: var(--raised); border: 1px solid var(--line);
    border-radius: 10px; padding: 14px 16px;
  }
  .stat .k { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
  .stat .v { font-size: 22px; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .stat .note { color: var(--muted); font-size: 12px; margin-top: 2px; font-variant-numeric: tabular-nums; }
  .callout {
    border: 1px solid var(--line); border-left: 3px solid var(--accent);
    background: var(--raised); border-radius: 8px; padding: 14px 18px; margin: 0 0 28px;
    font-size: 14px; color: var(--muted);
  }
  .callout strong { color: var(--ink); }
  .note-under { color: var(--muted); font-size: 13px; margin: 10px 2px 0; max-width: 92ch; }
  .sub { display: block; color: var(--muted); font-size: 12px; margin-top: 3px;
    font-variant-numeric: tabular-nums; font-weight: 400; }
  .wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; margin-top: 12px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
  th {
    background: var(--raised); color: var(--muted); font-size: 11.5px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .05em; white-space: nowrap;
  }
  tr:last-child td { border-bottom: none; }
  tr.here { background: var(--mark); }
  tr.here td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
  .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  a.name { color: var(--ink); font-weight: 650; text-decoration: none; border-bottom: 2px solid var(--accent); }
  a.name:hover { color: var(--accent); }
  .name { display: inline-block; }
  .formula { display: block; color: var(--muted); font-size: 12.5px; margin-top: 5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .desc { display: block; color: var(--muted); font-size: 13px; margin-top: 6px; max-width: 62ch; }
  .ratio { display: block; font-weight: 600; }
  .band { display: flex; align-items: center; gap: 6px; justify-content: flex-end;
    color: var(--muted); font-size: 12.5px; margin-top: 5px; }
  .band i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  .tag { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 4px;
    background: var(--raised); border: 1px solid var(--line); color: var(--muted); }
  .failed td { color: var(--muted); }
  .error { font-size: 13px; }
  .plot { margin: 0 -12px; }
  .backlink { display: inline-block; margin-bottom: 18px; color: var(--muted);
    font-size: 13px; text-decoration: none; }
  .backlink:hover { color: var(--accent); }
  footer { color: var(--muted); font-size: 13px; margin-top: 28px; max-width: 82ch; }
  footer p { margin: 0 0 10px; }
  footer strong { color: var(--ink); }
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<main>
{content}
</main>
</body>
</html>
"""


def _link(spec, highlight_key, current_page: str = "") -> str:
    """Model name, linked to its own chart unless we are already on it."""
    name = html.escape(spec.name)
    if spec.key == highlight_key:
        return f'<span class="name">{name}</span> <span class="tag">this chart</span>'
    return f'<a class="name" href="{spec.key}.html">{name}</a>'


def _row_class(spec, highlight_key) -> str:
    return ' class="here"' if spec.key == highlight_key else ""


def fit_table(results, highlight_key=None) -> str:
    """Fit quality and the complexity penalty, for every model."""
    rows = []
    for result in results:
        spec, m = result["spec"], result["metrics"]
        if m is None:
            rows.append(
                f'      <tr class="failed"><td>{html.escape(spec.name)}</td>'
                f'<td colspan="8" class="error">fit failed</td></tr>'
            )
            continue
        rows.append(
            f"""      <tr{_row_class(spec, highlight_key)}>
        <td>{_link(spec, highlight_key)}</td>
        <td class="num">{m.n_params}</td>
        <td class="num">{m.r2:.4f}</td>
        <td class="num">{m.adj_r2:.4f}</td>
        <td class="num">{m.sigma:.3f}</td>
        <td class="num">{m.medape:.1f}%</td>
        <td class="num">{m.d_aic:.2f}</td>
        <td class="num">{m.d_aicc:.2f}</td>
        <td class="num">{m.d_bic:.2f}</td>
      </tr>"""
        )
    return f"""  <div class="wrap"><table>
    <thead><tr>
      <th>Model</th><th class="num">k</th><th class="num">R²</th>
      <th class="num">adj R²</th><th class="num">σ</th><th class="num">MedAPE</th>
      <th class="num">ΔAIC</th><th class="num">ΔAICc</th><th class="num">ΔBIC</th>
    </tr></thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table></div>"""


def diagnostics_table(results, highlight_key=None) -> str:
    """Residual behaviour and how honest each model's bands are."""
    rows = []
    for result in results:
        spec, m = result["spec"], result["metrics"]
        if m is None:
            continue
        rows.append(
            f"""      <tr{_row_class(spec, highlight_key)}>
        <td>{_link(spec, highlight_key)}</td>
        <td class="num">{m.durbin_watson:.4f}</td>
        <td class="num">{m.rho1:.4f}</td>
        <td class="num">{m.skew:+.2f}</td>
        <td class="num">{m.excess_kurtosis:+.2f}</td>
        <td class="num">{m.coverage:.1f}%</td>
        <td class="num">{m.calibration_error:.1f}pp</td>
      </tr>"""
        )
    return f"""  <div class="wrap"><table>
    <thead><tr>
      <th>Model</th><th class="num">Durbin–Watson</th><th class="num">ρ₁</th>
      <th class="num">skew</th><th class="num">excess kurtosis</th>
      <th class="num">in rainbow</th><th class="num">calibration error</th>
    </tr></thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table></div>"""


def backtest_table(results, highlight_key=None) -> str:
    """Walk-forward forecast accuracy, ordered best first."""
    scored = [r for r in results if r.get("backtest")]
    if not scored:
        return ""

    ordered = sorted(scored, key=lambda r: r["backtest"].rmse_log)
    rows = []
    for position, result in enumerate(ordered):
        spec, b = result["spec"], result["backtest"]
        if position == 0:
            verdict = '<span class="tag">best</span>'
        elif b.tied_with_best:
            verdict = (
                f"[{b.rmse_delta_low:+.2f}, {b.rmse_delta_high:+.2f}] "
                '<span class="tag">tie</span>'
            )
        else:
            verdict = f"[{b.rmse_delta_low:+.2f}, {b.rmse_delta_high:+.2f}]"
        rows.append(
            f"""      <tr{_row_class(spec, highlight_key)}>
        <td>{_link(spec, highlight_key)}</td>
        <td class="num">{b.origins}</td>
        <td class="num">{b.rmse_log:.3f}</td>
        <td class="num">{b.bias_log:+.3f}</td>
        <td class="num">{b.medape:.1f}%</td>
        <td class="num">{b.coverage:.0f}%</td>
        <td class="num">{b.skill:+.2f}</td>
        <td class="num">{verdict}</td>
      </tr>"""
        )
    return f"""  <div class="wrap"><table>
    <thead><tr>
      <th>Model</th><th class="num">origins</th><th class="num">RMSE (log)</th>
      <th class="num">bias</th><th class="num">MedAPE</th><th class="num">in bands</th>
      <th class="num">skill vs RW</th><th class="num">vs best</th>
    </tr></thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table></div>"""


def stats_section(results, n_eff: float, n_hist: int, highlight_key=None) -> str:
    """The three comparison tables, covering every model."""
    fit_heading = (
        '<h2>Fit and complexity <span class="hint">— information criteria charged against '
        f"n_eff = {n_eff:.1f}, not {n_hist:,} rows</span></h2>"
    )
    fit_note = (
        '<p class="note-under">ΔAICc is the distance from the best model; 0 is best and a '
        "gap under about 2 is a tie. Adjusted R² is shown because it is the usual suggested "
        "remedy, but with n in the thousands and k ≤ 6 its correction lands in the fifth "
        "decimal — it does not separate these models.</p>"
    )
    diagnostics_heading = (
        '<h2>Residual diagnostics and band calibration <span class="hint">— are the bands '
        "honest?</span></h2>"
    )
    diagnostics_note = (
        '<p class="note-under">Durbin–Watson would be near 2 for independent residuals; '
        "these sit near 0, which is why the effective sample size above is a few dozen and "
        "not a few thousand. The σ bands assume log residuals are roughly normal, so skew "
        "shows up directly as calibration error — the mean gap between how much history a "
        "band actually holds and how much it should.</p>"
    )

    blocks = [
        fit_heading,
        fit_table(results, highlight_key),
        fit_note,
        diagnostics_heading,
        diagnostics_table(results, highlight_key),
        diagnostics_note,
    ]

    backtest = backtest_table(results, highlight_key)
    if backtest:
        backtest_heading = (
            '<h2>Out-of-sample predictive power <span class="hint">— expanding-window '
            "walk-forward, refitted at every origin</span></h2>"
        )
        backtest_note = (
            '<p class="note-under">bias &gt; 0 means the model under-predicted. Skill '
            "compares squared log error against a random walk, so positive beats it. The "
            "interval is a Bonferroni-widened moving-block bootstrap of the RMSE gap to the "
            "best model; where it straddles zero the two are statistically indistinguishable. "
            "Forecast windows overlap and there are only a couple of dozen, so read the "
            "ordering as indicative — it shifts when the evaluation window shifts.</p>"
        )
        blocks += [backtest_heading, backtest, backtest_note]
    return "\n".join(blocks)


def chart_page(plot_div: str, spec, results, n_eff: float, n_hist: int, heading=None) -> str:
    """A single model's chart, followed by the all-model comparison tables."""
    heading = heading or f"Bitcoin Rainbow Chart — {spec.name}"
    content = f"""  <a class="backlink" href="index.html">← all models</a>
  <h1>{html.escape(spec.name)}</h1>
  <p class="sub">{html.escape(spec.description)}</p>
  <div class="plot">{plot_div}</div>
{stats_section(results, n_eff, n_hist, highlight_key=spec.key)}
  <footer>
    <p>Every table above covers all models, with this one highlighted, so the chart can be
    read against its alternatives rather than on its own.</p>
    <p>A high R² means a model tracks history closely, not that it forecasts well — the most
    flexible curves overfit the past most eagerly. Nothing here is financial advice.</p>
  </footer>"""
    return PAGE.format(title=html.escape(heading), css=CSS, content=content)
