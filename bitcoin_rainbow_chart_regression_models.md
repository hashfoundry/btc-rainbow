# Bitcoin Rainbow Chart Regression Models

This document outlines the various logarithmic regression models used in Bitcoin rainbow charts, including their mathematical formulas and key characteristics.

## Introduction

Bitcoin rainbow charts are visual price prediction tools that use logarithmic regression to model Bitcoin's long-term price movements. These charts typically display colored bands that represent different market sentiments (from "bubble" to "fire sale") based on how far the price deviates from a central regression trend.

## Regression Model Variants

### 1. Power Law Corridor Model

**Formula:** 
```
P(t) = a × t^b
```

Where:
- P(t) = Bitcoin price at time t
- t = Time (often measured in days since Bitcoin's inception)
- a, b = Constants determined by regression analysis (b < 1 indicates diminishing returns)

**Corridor Bounds:**
```
Upper bound: P_upper(t) = a × t^b × (1 + c × σ)
Lower bound: P_lower(t) = a × t^b × (1 - c × σ)
```

Where:
- σ = Standard deviation of the regression
- c = Multiplier for determining band width

### 2. Log-Linear Regression Model

**Formula:**
```
log(P) = a × t + b
```

Which can be rewritten as:
```
P = e^(a×t + b)
```

Where:
- P = Bitcoin price
- t = Time (linear)
- a, b = Regression constants

**Rainbow Bands:**
```
Band_n(t) = e^(a×t + b + n×σ)
```

Where:
- n = Band position relative to center (e.g., -3, -2, -1, 0, 1, 2, 3)
- σ = Standard deviation of the log-linear regression

### 3. Log-Log Regression Model

**Formula:**
```
log(P) = a × log(t) + b
```

Which can be rewritten as:
```
P = t^a × e^b
```

Where:
- P = Bitcoin price
- t = Time
- a, b = Regression constants

**Applications:**
This model forms the basis for the Stock-to-Flow Cross Asset (S2FX) model when combined with additional parameters.

### 4. Hyperbolic Regression Model

**Formula:**
```
P(t) = a / (b - t)^c
```

Where:
- P(t) = Bitcoin price at time t
- t = Time
- a, b, c = Constants (b represents a future time point where the function would approach infinity)

### 5. Modified Power Law (Logarithmic) Regression

**Formula:**
```
log(P) = a × log(t)^c + b
```

Where:
- P = Bitcoin price
- t = Time
- a, b, c = Regression constants (c modifies the curvature of the regression line)

### 6. Logarithmic Regression with Fibonacci Bands

**Formula:**
```
Center band: log(P) = a × log(t) + b
Band_n: log(P_n) = a × log(t) + b ± n×f
```

Where:
- f = Fibonacci-scaled spacing factor
- n = Band number

### 7. Time-Adjusted Logarithmic Regression

**Formula:**
```
log(P) = a × log(t - t_offset) + b
```

Where:
- t_offset = Time offset parameter to adjust for Bitcoin's early price discovery phase

## Implementation Notes

- Most rainbow charts use 7-9 colored bands
- Band spacing can use equal divisions, standard deviations, or custom scaling factors
- Regression parameters are typically derived from Bitcoin's price history, often excluding extreme outliers
- Some implementations recalculate parameters periodically to adjust to evolving market behavior
- Halvings and market cycles may be incorporated as additional adjustment factors

## Limitations

All regression models are based on historical data and make assumptions about future behavior. They should be used as one of many tools for understanding Bitcoin's potential price movements rather than as definitive price predictions.
