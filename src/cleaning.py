"""
This script cleans the college scorecard data
"""
import pandas as pd
import numpy as np
import os

output_path = "data/college_scorecard_clean.csv"
cols = {
    # identifiers
    "UNITID":            "unitid",
    "INSTNM":            "name",
    "STABBR":            "state",
    "REGION":            "region",
    "CITY":              "city",
    # institution type
    "CONTROL":           "control",       # 1=public, 2=private nonprofit, 3=private for-profit
    "PREDDEG":           "pred_deg",      # 1=certificate, 2=associate, 3=bachelor's, 4=graduate
    "ICLEVEL":           "ic_level",      # 1=4-yr, 2=2-yr, 3=<2-yr
    "HBCU":              "hbcu",
    "HSI":               "hsi",
    "CURROPER":          "curr_oper",     # 1=currently operating
    # selectivity / student ability
    "ADM_RATE":          "adm_rate",
    "SAT_AVG":           "sat_avg",
    "SATVRMID":          "sat_vr_mid",
    "SATMTMID":          "sat_mt_mid",
    # cost / resources
    "TUITIONFEE_IN":     "tuition_in",
    "TUITIONFEE_OUT":    "tuition_out",
    "COSTT4_A":          "net_cost",
    "INEXPFTE":          "instruct_exp_fte",
    "AVGFACSAL":         "avg_fac_sal",
    "PFTFAC":            "pct_ft_fac",
    # student background (SES controls)
    "PCTPELL":           "pct_pell",
    "UGDS":              "enrollment",
    "MD_FAMINC":         "med_fam_inc",
    "PAR_ED_PCT_1STGEN": "pct_first_gen",
    # outcomes
    "MD_EARN_WNE_P6":    "md_earn_6yr",
    "MD_EARN_WNE_P10":   "md_earn_10yr",
    "C150_4":            "completion_rate",
    "RET_FT4":           "retention_rate",
}

def load_data():
    df = pd.read_csv(
        "data/college_scorecard_raw.csv",
        na_values=["NULL", "PrivacySuppressed"],
        low_memory=False,
    )
    df = df[list(cols.keys())].rename(columns=cols)
    return df

def coerce_numeric(df):
    str_cols = {"unitid", "name", "state", "city"}
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
 
 
def filter_universe(df):
    # Currently operating, 4-year schools, with a non-null earnings outcome
    df = df[(df["curr_oper"] == 1) & (df["ic_level"] == 1) & df["md_earn_10yr"].notna()]
    df = df.drop(columns=["curr_oper"])
    return df.reset_index(drop=True)
 
 
def consolidate_sat(df):
    # Use sat_avg when available; otherwise sum verbal + math midpoints
    df["sat_composite"] = df["sat_avg"].fillna(df["sat_vr_mid"] + df["sat_mt_mid"])
    df = df.drop(columns=["sat_avg", "sat_vr_mid", "sat_mt_mid"])
    return df
 
 
def consolidate_tuition(df):
    # Out-of-state for public schools, in-state for private; fill remaining gaps
    df["tuition"] = np.where(df["control"] == 1, df["tuition_out"], df["tuition_in"])
    df["tuition"] = df["tuition"].fillna(df["tuition_in"]).fillna(df["tuition_out"])
    df = df.drop(columns=["tuition_in", "tuition_out"])
    return df
 
 
def label_control(df):
    df["control"] = df["control"].map({1: "public", 2: "private_np", 3: "private_fp"})
    return df

def save(df, path):
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
 
 
def clean_data():
    df = load_data()
    df = coerce_numeric(df)
    df = filter_universe(df)
    df = consolidate_sat(df)
    df = consolidate_tuition(df)
    df = label_control(df)
    save(df, output_path)

