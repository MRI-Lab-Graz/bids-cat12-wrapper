# Design Matrix Configuration - Technical Note

## ⚠️ Critical: Well-Conditioned Design for Repeated Measures

This pipeline uses a **well-conditioned factorial design** for longitudinal/repeated measures data. Understanding this is crucial for valid statistical inference.

---

## The Problem: Ill-Conditioned Design

### ❌ WRONG Approach (Ill-Conditioned)
Including **both** subject factors AND repeated measures factors creates an ill-conditioned design matrix:

```matlab
% DON'T DO THIS!
Factor 1: Subject (124 levels)  % Identifies each individual
Factor 2: Group (3 levels)      % Between-subject
Factor 3: Time (3 levels)       % Within-subject
```

**Why this is wrong:**
- Subject and Group are perfectly confounded (each subject belongs to exactly one group)
- Creates rank-deficient design matrix
- SPM cannot estimate all parameters
- Results in invalid statistical inference

### Matrix Rank Problem
```
Subjects × Groups: 124 × 3 = 372 parameters
But: Only 124 subjects total (52 + 31 + 40 = 123)
Rank deficiency: Cannot estimate all subject effects AND group effects
```

---

## ✅ CORRECT Approach (Well-Conditioned)

### Our Design: Group × Time Only

```matlab
Factor 1: Group (3 levels, between-subject, dept=0)
  - control
  - intervention_2w
  - intervention_4w

Factor 2: Time (3 levels, within-subject, dept=1)
  - Session 1
  - Session 2
  - Session 3
```

**Why this works:**
- **Group** (dept=0): Independent between-subject factor
- **Time** (dept=1): Dependent within-subject factor (repeated measures)
- SPM automatically models within-subject dependencies
- No explicit subject factors needed
- Design matrix is full rank and estimable

---

## How SPM Handles Within-Subject Structure

### The `dept=1` Parameter

When you set `dept=1` for the Time factor, SPM:

1. **Recognizes repeated measures**: Knows that scans within the same timepoint are dependent
2. **Models dependencies implicitly**: Uses the cell structure to understand which scans come from the same subjects
3. **Adjusts standard errors**: Accounts for non-independence in the error structure
4. **No subject regressors needed**: Dependencies handled through variance modeling

### Cell Structure Example

```matlab
Cell 1: control × session_1    [52 scans from 52 different subjects]
Cell 2: control × session_2    [52 scans from SAME 52 subjects]
Cell 3: control × session_3    [52 scans from SAME 52 subjects]
Cell 4: intervention_2w × session_1  [31 scans from 31 different subjects]
Cell 5: intervention_2w × session_2  [31 scans from SAME 31 subjects]
...
```

SPM infers the subject structure from the **alignment** of scans across cells.

---

## Design Matrix Dimensions

### Our Well-Conditioned Design

```
Number of cells: 9 (3 groups × 3 timepoints)
Design matrix columns:
  - 9 cell means
  - Covariates (if specified)
  - Global effects (if any)
  
Total parameters: ~9-15 (depending on covariates)
Number of scans: 369
Degrees of freedom: 369 - parameters = ~354-360

✓ Full rank, well-conditioned, estimable
```

### Why This is Valid

1. **Each cell mean is estimable**: We have multiple scans per cell
2. **Contrasts work properly**: Can test group effects, time effects, interactions
3. **Within-subject variance properly modeled**: Through dept=1 specification
4. **Standard errors are correct**: Account for repeated measures correlation

---

## Statistical Model

### Implicit Model

The flexible factorial design with dept=1 effectively models:

```
Y_ijk = μ + α_i + β_j + (αβ)_ij + S_k(i) + ε_ijk

Where:
  Y_ijk = Observation for group i, time j, subject k
  μ     = Grand mean
  α_i   = Group effect (i = 1,2,3)
  β_j   = Time effect (j = 1,2,3)
  (αβ)_ij = Group × Time interaction
  S_k(i)  = Subject effect (random, nested in group)
  ε_ijk   = Error term
```

**Key point**: S_k(i) is **implicitly modeled** through the repeated measures structure (dept=1), NOT as explicit regressors.

---

## Contrasts That Work

With this design, you can test:

### ✅ Valid Contrasts

1. **Main effect of Group**
   ```matlab
   [1 0 -1 0 0 0 0 0 0]  % control vs intervention_4w (averaged over time)
   ```

2. **Main effect of Time**
   ```matlab
   [1 -1 0 1 -1 0 1 -1 0]  % session_1 vs session_2 (averaged over groups)
   ```

3. **Group × Time Interaction**
   ```matlab
   [(control_s1-control_s2) - (int2w_s1-int2w_s2)]
   ```

4. **Simple effects**
   ```matlab
   [1 -1 0 0 0 0 0 0 0]  % Time effect in control group only
   ```

### ❌ Invalid Contrasts

- Subject-specific effects (subjects not in design matrix)
- Between-subject within-group comparisons (not modeled)

---

## Verification

### Check Your Design Matrix

After estimation, check in MATLAB:

```matlab
load('SPM.mat');
rank(SPM.xX.X)        % Should equal number of columns
cond(SPM.xX.X)        % Condition number (lower is better)
                      % < 1000 is good, > 10000 is problematic
```

For well-conditioned design:
- `rank(SPM.xX.X) == size(SPM.xX.X, 2)` ✓ Full rank
- `cond(SPM.xX.X) < 1000` ✓ Well-conditioned

---

## References

1. **SPM Manual**: Chapter on Factorial Designs
   - https://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf

2. **Friston et al. (2007)**: Statistical Parametric Mapping
   - Details on flexible factorial designs

3. **Gläscher & Gitelman (2008)**: Contrast weights in flexible factorial designs
   - NeuroImage 41:1016-1027

4. **Key SPM Mailing List Posts**:
   - "Repeated measures without subject factor" (2010)
   - "dept parameter in factorial design" (2012)

---

## Summary

✅ **DO**: Use Group (dept=0) × Time (dept=1) design  
✅ **DO**: Let SPM handle within-subject structure implicitly  
✅ **DO**: Check design matrix rank and condition number  

❌ **DON'T**: Include subject factors explicitly  
❌ **DON'T**: Create ill-conditioned design matrices  
❌ **DON'T**: Ignore the dept parameter  

---

**Pipeline Implementation**: This design is automatically implemented in `utils/generate_spm_batch.py`

**Last Updated**: November 2025
