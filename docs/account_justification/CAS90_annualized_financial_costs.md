# CAS90: Annualized Financial Costs

**Date:** 2026-02-20
**Status:** Implemented

## What CAS90 Is

CAS90 converts the total capital investment into an equivalent annual payment
over the plant's operating lifetime. This is the "capital charge" component
of LCOE — the annual cost to investors for recovering their capital plus
return on investment.

## Account Structure

Per pyfecons/EMWG:

| CAS | Description | Status |
|-----|-------------|--------|
| 91  | Escalation — annual cost increases during operation (inflation, labor, materials) | Not implemented (constant-dollar model) |
| 92  | Fees — annual regulatory/licensing fees during operation | Not implemented |
| 93  | Cost of Money — capital charges (CRF-based, opportunity cost of capital) | Implemented |
| 99  | Contingency on annualized financial costs | Not implemented |

Our CAS90 currently computes only CAS93. CAS91 (operating-period
escalation) is partially handled by `levelized_annual_cost` inside
CAS70/CAS80, but that function is flagged as bug #6 in the critical
review — it inflates to operation-start dollars but does not properly
levelized a growing annuity over the plant lifetime.

### CAS91 vs CAS60

These are distinct:
- **CAS60** (CAS61/62/63): one-time capitalized costs incurred **during
  construction**, added to total capital investment before COD.
- **CAS91**: recurring annual escalation **during the 30-year operating
  life** — how O&M, fuel, and other annual costs grow year-over-year.

## Derivation

The Capital Recovery Factor converts a present value into equal annual
payments over $n$ years at interest rate $i$:

$$
\text{CRF}(i, n) = \frac{i(1+i)^n}{(1+i)^n - 1}
$$

The annualized financial cost is:

$$
\text{CAS90} = \text{CRF} \times \text{total\_capital}
$$

Where `total_capital` includes IDC (CAS60). At 7% WACC, 30-year plant life:
CRF = 0.0806, so CAS90 ~ 8.1% of total capital per year.

## Interaction with CAS60

**Critical design decision:** CAS60 and CAS90 must use complementary
conventions to avoid double-counting construction-period financing.

### The double-counting bug (prior to fix)

The previous implementation used an "effective CRF" that included a
`(1+i)^T` multiplier:

```python
def compute_effective_crf(interest_rate, plant_lifetime, construction_time):
    crf = compute_crf(interest_rate, plant_lifetime)
    return crf * (1 + interest_rate) ** construction_time

def cas90_financial(cc, total_capital, ...):
    eff_crf = compute_effective_crf(interest_rate, plant_lifetime, t_project)
    return eff_crf * total_capital
```

The `(1+i)^T` factor shifts the capital cost from start-of-construction
to COD — effectively computing IDC assuming a lump-sum disbursement at
time zero. But `total_capital` already included CAS60 (which computes IDC
assuming uniform spending). This double-counted construction financing:

```
total_capital = overnight + CAS60        # CAS60 adds IDC
CAS90 = CRF * (1+i)^T * total_capital   # (1+i)^T adds IDC AGAIN
```

At 7% WACC, 6-year construction, the `(1+i)^T` factor is 1.50 — so the
old code overstated capital charges by ~50%.

### The two valid conventions

**Option A (implemented): Explicit IDC + plain CRF**
```
overnight = CAS10 + CAS20 + CAS30 + CAS40 + CAS50
CAS60 = f_IDC * overnight
total_capital = overnight + CAS60
CAS90 = CRF * total_capital
```

**Option B (not used): No IDC + effective CRF**
```
total_capital = CAS10 + CAS20 + CAS30 + CAS40 + CAS50
CAS60 = 0
CAS90 = CRF * (1+i)^T * total_capital
```

We chose Option A because:
1. IDC is visible as a separate line item (transparency)
2. Uniform spending (CAS60's f_IDC) is more realistic than lump-sum
   ((1+i)^T) — the difference is ~26% at reference conditions
3. Standard practice in EMWG/ARIES cost reporting

### Numerical comparison at reference conditions

| Convention | CAS60 | CAS90 | Annual total |
|-----------|-------|-------|-------------|
| Old (double-counted) | $0.55M* | $605M | $605M |
| Option A (implemented) | $961M | $480M | $480M |
| Option B (lump-sum) | $0 | $605M | $605M |

*CAS60 was also broken by /1e3 bug, so the double-counting was masked.

## Source Analysis

### pyfecons

pyfecons computes CAS90 using the effective CRF, and also computes CAS60
separately. However, pyfecons' CAS60 was so small relative to total capital
(due to using `p_net * 0.099 * time` instead of compound interest on capital)
that the double-counting had limited practical impact.

### Standard practice

GEN-IV EMWG and ARIES reports consistently show CAS60 as a separate line
item in capital cost summaries, implying they use plain CRF for
annualization. The effective CRF appears in some simplified LCOE formulas
that omit CAS60 — it is an alternative, not a complement.

## Implementation

```python
def cas90_financial(total_capital, interest_rate, plant_lifetime):
    crf = compute_crf(interest_rate, plant_lifetime)
    return crf * total_capital
```

Removed:
- `compute_effective_crf` from `economics.py` (prevents future double-counting)
- `cc`, `construction_time`, `fuel`, `noak` parameters (no longer needed)

## References

- GEN-IV EMWG, "Cost Estimating Guidelines for Generation IV Nuclear
  Energy Systems," Rev. 4.2, 2007. (Account 90 definition, CRF formula)
- Schulte et al., "Fusion Reactor Design Studies — Standard Accounts
  for Cost Estimates," PNL-2648, 1978. (Original account structure)
- Any engineering economics textbook (CRF derivation)
