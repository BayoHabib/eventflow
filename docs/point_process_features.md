# Point-Process Feature Engineering

This document explains the point-process features in EventFlow. These are **feature engineering steps**, not model fitting.

## Key Principle

All point-process steps compute features using **user-provided parameters**. No model fitting occurs - the downstream ML model learns how to use these features.

```python
# Parameters are specified, not learned
step = HawkesKernelStep(
    mu=0.1,      # User provides
    alpha=0.5,   # User provides  
    beta=1.0,    # User provides
)
```

---

## Hawkes Process Features

### The Formula

The Hawkes conditional intensity at time $t$ is:

$$\lambda(t) = \mu + \sum_{t_i < t} \alpha \cdot e^{-\beta(t - t_i)}$$

Where:
- $\mu$ = baseline rate (background intensity)
- $\alpha$ = excitation magnitude (how much each event triggers)
- $\beta$ = decay rate (how fast excitation fades)
- $t_i$ = past event times

### What Gets Computed

| Output Column | Formula | Meaning |
|---------------|---------|---------|
| `hawkes_intensity` | $\lambda(t)$ | Total intensity at event time |
| `hawkes_background` | $\mu$ | Baseline component |
| `hawkes_trigger` | $\sum \alpha e^{-\beta \Delta t}$ | Excitation from past events |

### No Fitting Required

```python
def compute_intensity(current_time, past_event_times, mu, alpha, beta):
    """Pure computation - no fitting."""
    trigger_sum = 0.0
    for t_i in past_event_times:
        dt = current_time - t_i
        trigger_sum += alpha * math.exp(-beta * dt)
    
    return mu + trigger_sum
```

---

## Parameter Sources

Since parameters are not learned, where do they come from?

| Source | Description | Example |
|--------|-------------|---------|
| **Domain knowledge** | Expert understanding of the phenomenon | "Crime clusters within 24h" → β = 1/24 |
| **Literature** | Published research values | Repeat victimization studies |
| **Prior offline fitting** | Fit a Hawkes model once, extract params | MLE on historical data |
| **Hyperparameter tuning** | Treat as hyperparams for downstream model | Grid search α, β via cross-validation |
| **Sensitivity analysis** | Try multiple values, compare feature importance | β ∈ {0.5, 1.0, 2.0} |

---

## Available Steps

### Decay & Recency

| Step | Output | Description |
|------|--------|-------------|
| `ExponentialDecayStep(beta)` | `decay_weight` | Time-weighted recency: $e^{-\beta \Delta t}$ |
| `ContinuousInterEventStep()` | `inter_event_seconds` | Precise inter-event times |

### Hawkes Process

| Step | Output | Description |
|------|--------|-------------|
| `HawkesKernelStep(mu, alpha, beta)` | `hawkes_*` | Intensity decomposition |
| `ConditionalIntensityStep(mu, alpha, beta)` | `conditional_intensity` | Point intensity estimate |

### Survival Analysis

| Step | Output | Description |
|------|--------|-------------|
| `HazardRateStep(baseline_hazard)` | `hazard_rate` | Instantaneous risk: $h(t) = \frac{f(t)}{S(t)}$ |
| `SurvivalTableStep(time_horizons)` | `survival_*` | $P(\text{no event in } [t, t+h])$ |
| `DurationFeaturesStep()` | `duration_*` | Gap statistics |

### Spatial Point-Process

| Step | Output | Description |
|------|--------|-------------|
| `PairCorrelationStep(max_distance)` | `pair_correlation` | Spatial clustering $g(r)$ |
| `KFunctionStep(max_distance, window=...)` | `k_function_local`, optional `l_function_local` / `h_function_local` | Window-normalized local K/L/H at one radius |

---

## Local K/L/H contract

`KFunctionStep` requires a caller-declared `SpatialDomain` observation window in
a projected CRS measured in meters. Coordinates are transformed into that CRS
for computation without changing the input coordinate columns. All records must
be finite and lie in the window; outside rows are never silently discarded.
The step treats its input as one declared point pattern. Select the intended
time interval/type beforehand. Window geometry alone does not certify that
observation of the phenomenon is complete.

For n >= 2, the local statistic is

`K_i(r) = area(W)/(n-1) * sum(j != i, d_ij <= r, e_ij)`.

The default `edge_correction="translation"` uses
`e_ij = area(W)/area(W intersection (W + displacement_ij))`.
Holes and disjoint window parts participate in both the area and intersection.
Zero overlap raises explicitly. `edge_correction="none"` uses unit weights;
this has edge bias and should be an explicit analytic choice. Ripley/isotropic
and border corrections are not implemented and are rejected, including the old
misleading default `"ripley"`.

`L_i=sqrt(K_i/pi)` and `H_i=L_i-r` are applied consistently, including K_i=0.
K has square-meter units; L and H have meter units. The mean of local K values
is the corresponding global K estimator, but averaging transformed local L/H
values is **not** the global L/H transform. Fewer than two points gives typed
null statistics. Empty input retains typed empty feature columns. Coincident
records count as distinct records; deduplication is a separate quality policy.

This homogeneous estimator assumes a stationary point process. No significance
or predictive calibration is inferred from H's sign. Inhomogeneous estimation
and simulation envelopes remain separate work. Single-radius computation and
translation intersections materialize the input and can take quadratic time;
this is not a streaming or large-scale point-pattern engine. The old unused
`n_bins` parameter is rejected; repeat the step over explicit radii to build a
curve, keeping each result distinct.

See the [migration guide and analytic references](real_data_audit_fixes.md) for
examples, compatibility changes and the real-data replay command.

---

## Streaming Variants

For online/incremental computation:

| Step | Description |
|------|-------------|
| `StreamingHawkesStep` | Online Hawkes intensity updates |
| `OnlineStatisticsStep` | Running mean/variance/count |
| `StreamingInterEventStep` | Online inter-event tracking |
| `EventBufferStep` | Maintains event history window |

---

## Example Pipeline

```python
from eventflow.core.pipeline import Pipeline
from eventflow.core.steps import (
    InterEventTimeStep,
    ExponentialDecayStep,
    HawkesKernelStep,
    HazardRateStep,
)

pipeline = Pipeline(steps=[
    # Basic temporal features
    InterEventTimeStep(),
    
    # Point-process features with fixed parameters
    ExponentialDecayStep(beta=1.0),
    HawkesKernelStep(mu=0.1, alpha=0.5, beta=1.0),
    HazardRateStep(baseline_hazard=0.05),
])

# Output: EventFrame with new feature columns
result = pipeline.run(event_frame)

# Feed to downstream ML model
features = result.collect().select([
    "decay_weight",
    "hawkes_intensity", 
    "hawkes_trigger",
    "hazard_rate",
])
```

---

## Why This Design?

### Feature Engineering vs Modeling

| Aspect | Feature Engineering (EventFlow) | Modeling |
|--------|--------------------------------|----------|
| Parameters | User-specified | Learned from data |
| Output | Feature columns | Predictions / likelihood |
| Goal | Enrich data for ML | Inference / forecasting |
| Fitting | None | MLE, Bayesian, etc. |

### Benefits

1. **Reproducibility** - Same params → same features
2. **Speed** - No iterative optimization
3. **Flexibility** - Downstream model decides feature importance
4. **Composability** - Combine with other feature steps
5. **Interpretability** - Each feature has clear meaning

---

## Validation Utilities

Separate from steps, validation functions check feature quality:

```python
from eventflow.core.steps.validation import (
    validate_intensity_positivity,
    validate_hawkes_stability,
    validate_survival_monotonicity,
)

# Check Hawkes stability: α/β < 1
result = validate_hawkes_stability(alpha=0.5, beta=1.0)
assert result.is_valid  # True, branching ratio = 0.5

# Check intensity is non-negative
result = validate_intensity_positivity(event_frame, "hawkes_intensity")
```

---

## References

- Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually exciting point processes."
- Reinhart, A. (2018). "A review of self-exciting spatio-temporal point processes and their applications."
- Mohler, G. O., et al. (2011). "Self-exciting point process modeling of crime."
