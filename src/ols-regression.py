"""
This script runs the OLS Regression of Median Graduate Earnings on several characteristics

- Earnings: md_earn_10yr (median earnings 10 years post-enrollments

- Characteristics: adm_rate (admissionn rate), sat_composite (SAT score), tuition, 
pct_pell (% of Pell Grants), control (institution type)

- Fixed Effects: state dummies to control for state-level differences in earnings
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

input_path  = "data/college_scorecard_clean.csv"
output_path = "data/ols_results.txt"

# 1. Load cleaned data
df = pd.read_csv(input_path)

# 2. Variable selection & sample
X = ["adm_rate", "sat_composite", "tuition", "control", "pct_pell"]
Y    = "md_earn_10yr"

keep = [Y] + X + ["state"] #sample creation
df = df[keep].dropna() #drops rows with missing values 

print(f"Observations in estimation sample: {len(df)}") 

# 3. OLS formula with state fixed effects
# C(control) creates dummies for institution type (base = private_fp) to isolate effect of ind. vars
# C(state) creates state fixed effects (only comparing schools within the same state)
formula = (
    "md_earn_10yr ~ adm_rate + sat_composite + tuition "
    "+ C(control) + pct_pell + C(state)"
)

model  = smf.ols(formula=formula, data=df)
result = model.fit(cov_type="HC3")   # heteroskedasticity-robust SEs

# 4. Print results
# Show only the main regressors (suppress state FE rows for readability)
main_vars = [v for v in result.params.index if not v.startswith("C(state)")]

print("\n" + "="*65)
print("OLS: Median 10-Year Earnings on Admission Rate, SAT, Tuition, Pell Grants, and Institution Type")
print("State Fixed Effects included  |  Robust (HC3) SEs")
print("="*65)
print(f"{'Variable':<30} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
print("-"*65)
for v in main_vars:
    coef = result.params[v]
    se   = result.bse[v]
    t    = result.tvalues[v]
    p    = result.pvalues[v]
    stars = ("***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "")
    print(f"{v:<30} {coef:>10.1f} {se:>10.1f} {t:>8.2f} {p:>8.3f} {stars}")

print("-"*65)
print(f"{'N':<30} {result.nobs:>10.0f}")
print(f"{'R-squared':<30} {result.rsquared:>10.4f}")
print(f"{'Adj. R-squared':<30} {result.rsquared_adj:>10.4f}")
print("="*65)
print("* p<0.10  ** p<0.05  *** p<0.01")

# 5. Save full results to text file
with open(output_path, "w") as f:
    f.write(result.summary().as_text())
print(f"\nFull results saved to {output_path}")