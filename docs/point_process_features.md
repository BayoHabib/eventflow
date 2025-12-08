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
| `KFunctionStep(max_distance)` | `k_function` | Ripley's K statistic |

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
