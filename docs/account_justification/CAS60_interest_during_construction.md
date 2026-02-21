# CAS60: Capitalized Financial Costs (Interest During Construction)

**Date:** 2026-02-20
**Status:** Implemented

## What IDC Is

Interest During Construction (IDC) represents the cost of financing a power
plant during the construction period. Capital is spent before revenue begins,
so each dollar spent accrues interest from the time it is disbursed until
commercial operation date (COD). IDC is a capitalized cost — it increases the
total capital investment but does not correspond to physical equipment.

## Account Structure

Per pyfecons/EMWG:

| CAS | Description | Status |
|-----|-------------|--------|
| 61  | Escalation during construction | Not implemented (see below) |
| 62  | Fees paid during construction | Not implemented |
| 63  | Interest during construction (IDC proper) | Implemented |

### CAS61 — Escalation during construction

pyfecons computes CAS61 as:
```python
C610000 = n_mod * p_nrl / a_power * a_c_98
```
where `a_c_98 = 115` and `a_power = 1000` (both labeled `Unknown` with
`# TODO what are these?` in the pyfecons source). At reference conditions
this gives ~$299M. pyfecons' own comments state "f_EDC for constant-dollar
costing is zero" (cas60_capitalized_financial.py line 37), yet the code
computes it anyway using undocumented coefficients.

In a constant-dollar model, escalation should be zero — cost increases due
to general inflation are removed by convention. Real escalation (above
inflation) could be non-zero but is speculative and typically excluded from
initial estimates (as pyfecons' CAS900000.tex acknowledges for CAS91).

We do not implement CAS61. If real escalation becomes a modeling priority,
it should be added with documented, defensible coefficients.

## Derivation

### Uniform spending assumption

If total overnight cost $C$ is spent uniformly over $T$ years at annual
interest rate $i$, each year's increment $C/T$ accrues compound interest
from its disbursement date to COD. Using end-of-year disbursement:

$$
\text{FV} = \frac{C}{T} \sum_{k=0}^{T-1} (1+i)^k
           = \frac{C}{T} \cdot \frac{(1+i)^T - 1}{i}
$$

IDC is the excess above the overnight cost:

$$
\text{IDC} = \text{FV} - C = C \left[ \frac{(1+i)^T - 1}{iT} - 1 \right]
$$

Define the IDC fraction:

$$
f_{\text{IDC}}(i, T) = \frac{(1+i)^T - 1}{iT} - 1
$$

### Continuous spending variant

If spending is continuous rather than discrete annual increments:

$$
f_{\text{IDC}}^{\text{cont}}(i, T) = \frac{e^{iT} - 1}{iT} - 1
$$

This approximately matches the ARIES-II/IV Table 2.2-XVII values reproduced
in pyfecons comments (but not used in its calculations):

| T (yr) | f_IDC (5%) | f_IDC (2%) |
|--------|-----------|-----------|
| 3      | 0.079     | 0.031     |
| 4      | 0.106     | 0.041     |
| 6      | 0.163     | 0.061     |
| 8      | 0.221     | 0.082     |
| 10     | 0.284     | 0.105     |
| 12     | 0.349     | 0.127     |

### Convention choice

We use the **discrete end-of-year** convention. At reference conditions
(7% WACC, 6-year construction):

- Discrete annual: f_IDC = 0.192
- Continuous: f_IDC = 0.243

The discrete convention is more conservative (~21% lower) and consistent
with the model's other annual time-stepping assumptions. The difference
is well within overall model uncertainty.

## Interaction with CAS90

**Critical:** CAS60 and CAS90 must use complementary conventions to avoid
double-counting construction financing.

- **CAS60** adds explicit IDC to the capital cost: `total_capital = overnight + IDC`
- **CAS90** annualizes `total_capital` using **plain CRF** (not effective CRF)

See `CAS90_annualized_financial_costs.md` for full details on the
double-counting issue and its resolution.

## Source Analysis

### Where IDC lives in pyfecons vs 1costingfe

pyfecons and 1costingfe put the interest-rate-based IDC computation in
**different accounts**:

**pyfecons**: IDC is in **CAS90** via the effective CRF.
```python
# cas90_annualized_financial.py
effective_crf = CRF * (1 + interest_rate) ** construction_time
cas90.C900000 = effective_crf * total_capital
```
The `(1+i)^Tc` factor is where pyfecons actually computes compound interest
on capital during construction. pyfecons CAS60/CAS63 (`p_net * 0.099 * time`)
is a separate empirical formula with no connection to interest rates — it is
stacked on top of the effective CRF, resulting in double-counting.

**1costingfe**: IDC is in **CAS60** via the f_IDC formula.
```python
# cas60
f_idc = ((1 + i)**T - 1) / (i * T) - 1
CAS60 = f_idc * overnight_cost
# cas90
CAS90 = CRF * total_capital   # plain CRF, no (1+i)^Tc
```
IDC is an explicit capital cost line item. CAS90 uses plain CRF with no
construction-time adjustment. No double-counting.

### pyfecons CAS60 detail

pyfecons CAS60 has three components:

1. **CAS61 — Escalation** (actually computed):
   ```python
   C610000 = n_mod * p_nrl / a_power * a_c_98   # ~$299M at reference
   ```
   Undocumented coefficients (`a_c_98=115`, `a_power=1000`, both labeled
   `Unknown` with `# TODO what are these?`). Contradicts pyfecons' own
   comment that f_EDC = 0 for constant-dollar costing.

2. **CAS63 — IDC bottom-up** (used in total):
   ```python
   C630000 = p_net * idc_coeff * t_project   # $594M at reference
   ```
   Linear in power and time. Not interest-rate-based. The `idc_coeff = 0.099`
   is empirically calibrated, not derived from financial parameters.

3. **CAS63 — IDC via LSA** (computed but not used in total):
   ```python
   C630000LSA = fac_97[lsa-1] * C200000   # ~$1,060M at reference
   ```
   Fraction of total direct cost from Miller/ARIES lookup table.

pyfecons total: `C600000 = C630000 + C610000 = $594M + $299M = $893M`.
Plus the `(1+i)^Tc` factor in CAS90's effective CRF adds another layer
of construction-period financing on top.

### 1costingfe (prior to fix)

```python
def cas60_idc(cc, cas20, p_net, construction_time, fuel, noak):
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    return p_net * cc.idc_coeff * t_project / 1e3
```

Four compounding errors:
1. `/1e3` spurious divisor (same bug pattern as CAS30)
2. `idc_coeff = 0.05` instead of deriving from interest rate
3. Uses `p_net` instead of capital cost — IDC has no direct relationship
   to electrical output
4. `cas20` accepted but never used (dead parameter)

### Numerical comparison (1 GWe DT tokamak, ~$5B overnight)

| Method | CAS60 | CAS90 IDC component | Total IDC effect |
|--------|-------|--------------------|----|
| 1costingfe (broken) | $0.55M | ~$2,500M via eff. CRF | double-counted, broken |
| pyfecons | $893M | ~$2,500M via eff. CRF | double-counted |
| 1costingfe (fixed) | $961M | $0 (plain CRF) | $961M, no double-count |

## Implementation

```python
def cas60_idc(interest_rate, overnight_cost, construction_time):
    i = interest_rate
    T = construction_time
    f_idc = ((1 + i)**T - 1) / (i * T) - 1
    return f_idc * overnight_cost
```

Call site computes overnight cost explicitly:
```python
overnight_cost = c10 + c20 + c30 + c40 + c50
c60 = cas60_idc(interest_rate, overnight_cost, construction_time_yr)
total_capital = overnight_cost + c60
```

Removed from CostingConstants: `idc_coeff` (no longer needed — IDC is
derived from `interest_rate`, already a model input at 7% default).

Removed from economics.py: `compute_effective_crf` (prevents future
double-counting; see CAS90 justification).

## References

- GEN-IV EMWG, "Cost Estimating Guidelines for Generation IV Nuclear
  Energy Systems," Rev. 4.2, 2007. (Account 60 definition)
- Schulte et al., "Fusion Reactor Design Studies — Standard Accounts
  for Cost Estimates," PNL-2648, 1978. (Original account structure)
- Miller, "ARIES-RS Tokamak Power Plant — Power Core and Maintenance,"
  Fusion Engineering and Design, 1997. (LSA factor table, f_IDC table)
- pyfecons `cas60_capitalized_financial.py` (CAS60 computation)
- pyfecons `cas90_annualized_financial.py` (effective CRF — where pyfecons
  puts its interest-rate-based IDC)
- pyfecons `pycosting_arpa_e_mfe.py` lines 2103-2132 (f_IDC table, unused)
