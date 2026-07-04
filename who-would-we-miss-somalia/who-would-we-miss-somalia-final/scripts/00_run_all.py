"""
Who Would We Miss? Bayesian Climate Shock-to-Action Targeting Analysis

Run from repository root after placing official SIHBS 2022 .dta files in data/raw/.
Raw microdata and household-level predictions should not be committed publicly.
"""
from pathlib import Path
import json, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
warnings.filterwarnings("ignore")

RAW = Path("data/raw")
OUT = Path("outputs"); FIG = Path("figures"); META = Path("data/metadata"); PROC = Path("data/processed")
for p in [OUT, FIG, META, PROC]: p.mkdir(parents=True, exist_ok=True)

def read_dta(name, cats=True, cols=None):
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Place official SIHBS .dta files in data/raw/.")
    return pd.read_stata(path, convert_categoricals=cats, columns=cols)

def yes(s):
    return s.astype(str).str.strip().str.lower().eq("yes").astype(int)

def any_yes(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].apply(lambda x: x.astype(str).str.strip().str.lower().eq("yes")).any(axis=1).astype(int)

def wmean(x, w):
    x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
    return float(np.nansum(x*w) / np.nansum(w))

def wsum(x, w):
    return float(np.nansum(np.asarray(x, dtype=float) * np.asarray(w, dtype=float)))

# 1) Module inventory
inventory=[]
for f in sorted(RAW.glob("*.dta")):
    tmp = pd.read_stata(f, convert_categoricals=False)
    inventory.append({"file": f.name, "rows": tmp.shape[0], "columns": tmp.shape[1]})
pd.DataFrame(inventory).to_csv(META/"sihbs_modules_inventory.csv", index=False)

# 2) Household core and food-security experience proxy
food_cols = [f"foodsec7_0{i}" for i in range(2, 9)] + ["foodsec7_9"]
hh_cols = ["hid","hhsize","wgt","wgt_adj","region_n","ea_type_n","hh_water_type","hh_sewer_type","hh_sanitary_excl"] + food_cols
hh = read_dta("hh.dta", cats=True, cols=hh_cols)
hh["hid"] = hh["hid"].astype(int)
hh["region"] = hh["region_n"].astype(str)
hh["residence"] = hh["ea_type_n"].astype(str)
hh["weight"] = hh["wgt_adj"].fillna(hh["wgt"]).astype(float)
hh["fies_raw_score"] = sum(yes(hh[c]) for c in food_cols)
hh["moderate_severe_food_insecurity"] = (hh["fies_raw_score"] >= 4).astype(int)
hh["severe_food_insecurity"] = (hh["fies_raw_score"] >= 7).astype(int)
water = hh["hh_water_type"].astype(str).str.lower()
improved = ["piped water into dwelling","piped water to yard","public tap","standpipe","tubewell","borehole","protected dug well","protected spring","rainwater"]
hh["improved_water"] = water.apply(lambda v: any(p in v for p in improved)).astype(int)
san = hh["hh_sewer_type"].astype(str).str.lower()
unimproved = ["no toilet", "bush", "field", "open pit latrine", "bucket"]
hh["unimproved_sanitation"] = san.apply(lambda v: any(p in v for p in unimproved)).astype(int)
hh["wash_deprivation"] = ((hh["improved_water"] == 0) | (hh["unimproved_sanitation"] == 1)).astype(int)

# 3) Poverty/IDP, head characteristics, shocks, and livelihood proxies
cons = read_dta("consagg.dta", cats=True, cols=["hid","poor","pcer_dec","idp","pcer"])
cons["hid"] = cons["hid"].astype(int)
cons["poor"] = (cons["poor"].astype(str).str.lower().str.contains("poor") & ~cons["poor"].astype(str).str.lower().str.contains("non")).astype(int)
cons["idp"] = cons["idp"].astype(str).str.lower().eq("yes").astype(int)

hhm = read_dta("hhm.dta", cats=True, cols=["hid","pid","qsn1_03","qsn1_04","qsn2_03","qsn2_04","qsn2_05"])
hhm["hid"] = hhm["hid"].astype(int)
head = hhm[hhm["qsn1_04"].astype(str).eq("Head")].drop_duplicates("hid")
head = head[["hid","qsn1_03","qsn2_03","qsn2_04","qsn2_05"]].copy()
head["female_headed"] = head["qsn1_03"].astype(str).str.lower().eq("female").astype(int)
head["head_literate"] = (head["qsn2_03"].astype(str).str.lower().eq("yes") | head["qsn2_04"].astype(str).str.lower().eq("yes")).astype(int)
head["head_ever_school"] = head["qsn2_05"].astype(str).str.lower().eq("yes").astype(int)
head = head[["hid","female_headed","head_literate","head_ever_school"]]

shock = read_dta("shock.dta", cats=True)
shock["hid"] = shock["hid"].astype(int)
stype = shock["shock10_03"].astype(str)
shock["has_shock_row"] = (~stype.str.lower().isin(["nan",""])).astype(int)
shock["drought_shock"] = stype.str.contains("drought|water shortage", case=False, na=False).astype(int)
shock["food_price_shock"] = stype.str.contains("price for food|rise in price for food", case=False, na=False).astype(int)
shock["livestock_death_shock"] = stype.str.contains("livestock died", case=False, na=False).astype(int)
shock["flood_shock"] = stype.str.contains("flood", case=False, na=False).astype(int)
shock["crop_shock"] = stype.str.contains("crop disease|crop pests|fall in sale prices for crops|agric. input prices|agricultural input", case=False, na=False).astype(int)
formal_cols = ["shock10_07_17","shock10_07_18","shock10_07_19"]
erosive_cols = ["shock10_07_3","shock10_07_4","shock10_07_5","shock10_07_6","shock10_07_11","shock10_07_12","shock10_07_14","shock10_07_21","shock10_07_22","shock10_07_23"]
shock["formal_support_received_row"] = any_yes(shock, formal_cols)
shock["erosive_coping_row"] = any_yes(shock, erosive_cols)
shock["any_income_asset_loss"] = shock["shock10_04"].astype(str).str.lower().str.contains("income|asset|both").fillna(False).astype(int)
sh = shock.groupby("hid").agg(shock_exposed=("has_shock_row","max"), drought_shock=("drought_shock","max"), food_price_shock=("food_price_shock","max"), livestock_death_shock=("livestock_death_shock","max"), flood_shock=("flood_shock","max"), crop_shock=("crop_shock","max"), formal_support_received=("formal_support_received_row","max"), erosive_coping=("erosive_coping_row","max"), any_income_asset_loss=("any_income_asset_loss","max")).reset_index()

liv = read_dta("livestock_own.dta", cats=True, cols=["hid","liv4_02","liv4_03"]); liv["hid"] = liv["hid"].astype(int)
liv["livestock_owner_row"] = (yes(liv["liv4_02"]) | yes(liv["liv4_03"])).astype(int)
liv = liv.groupby("hid").agg(livestock_owner=("livestock_owner_row","max")).reset_index()
crops = read_dta("crops.dta", cats=True, cols=["hid","crop6_21"]); crops["hid"] = crops["hid"].astype(int)
crops["crop_activity_row"] = yes(crops["crop6_21"])
crops = crops.groupby("hid").agg(crop_activity=("crop_activity_row","max")).reset_index()
bus = read_dta("hhbus.dta", cats=True, cols=["hid","hb8_05"]); bus["hid"] = bus["hid"].astype(int)
bus = bus.assign(business_activity=1).groupby("hid")["business_activity"].max().reset_index()

# 4) Analysis-ready dataset
df = hh.merge(cons[["hid","poor","idp"]], on="hid", how="left")
for m in [head, sh, liv, crops, bus]:
    df = df.merge(m, on="hid", how="left")
for c in ["poor","idp","female_headed","head_literate","head_ever_school","shock_exposed","drought_shock","food_price_shock","livestock_death_shock","flood_shock","crop_shock","formal_support_received","erosive_coping","any_income_asset_loss","livestock_owner","crop_activity","business_activity"]:
    df[c] = df[c].fillna(0).astype(int)
df["climate_livelihood_shock"] = (df[["drought_shock","food_price_shock","livestock_death_shock","flood_shock","crop_shock"]].sum(axis=1) > 0).astype(int)
df["high_risk_unreached"] = ((df["climate_livelihood_shock"] == 1) & ((df["moderate_severe_food_insecurity"] == 1) | (df["erosive_coping"] == 1)) & (df["formal_support_received"] == 0)).astype(int)
df["direct_need_score"] = df["moderate_severe_food_insecurity"] + df["erosive_coping"] + df["any_income_asset_loss"] + df["severe_food_insecurity"] + df["climate_livelihood_shock"]
df["hhsize_z"] = (df["hhsize"] - df["hhsize"].mean()) / df["hhsize"].std(ddof=0)
df["region"] = df["region"].astype("category")
df["residence"] = df["residence"].astype("category")
# Public schema only, no household rows
pd.DataFrame({"variable": list(df.columns)}).to_csv(PROC/"analysis_ready_schema_only.csv", index=False)

# 5) Bayesian hierarchical logistic model
formula = "high_risk_unreached ~ drought_shock + food_price_shock + livestock_death_shock + flood_shock + crop_shock + C(residence) + hhsize_z + female_headed + poor + idp + wash_deprivation + livestock_owner + crop_activity + business_activity + head_literate"
fit = BinomialBayesMixedGLM.from_formula(formula, {"region": "0 + C(region)"}, df, vcp_p=1, fe_p=2).fit_vb(verbose=False, minim_opts={"maxiter": 1000})
df["bayesian_risk"] = fit.predict()
fe = pd.DataFrame({"term": fit.model.exog_names, "posterior_mean_log_odds": fit.fe_mean, "posterior_sd": fit.fe_sd})
fe["odds_ratio"] = np.exp(fe["posterior_mean_log_odds"])
fe["or_2.5"] = np.exp(fe["posterior_mean_log_odds"] - 1.96*fe["posterior_sd"])
fe["or_97.5"] = np.exp(fe["posterior_mean_log_odds"] + 1.96*fe["posterior_sd"])
fe.to_csv(OUT/"posterior_predictor_effects.csv", index=False)

# 6) Weighted summaries and targeting simulation
summary_vars = [("Climate/livelihood shock exposure","climate_livelihood_shock"),("Drought/severe water shortage shock","drought_shock"),("Food-price shock","food_price_shock"),("Moderate/severe food insecurity proxy","moderate_severe_food_insecurity"),("Formal support received after shock","formal_support_received"),("High-risk unreached outcome","high_risk_unreached")]
pd.DataFrame([{"indicator": lab, "weighted_percent": 100*wmean(df[var], df["weight"])} for lab,var in summary_vars]).to_csv(OUT/"weighted_descriptive_summary.csv", index=False)
region = df.groupby("region").apply(lambda g: pd.Series({"sample_n": len(g), "weighted_high_risk_unreached_percent": 100*wmean(g["high_risk_unreached"], g["weight"]), "mean_bayesian_predicted_risk_percent": 100*wmean(g["bayesian_risk"], g["weight"])})).reset_index().sort_values("mean_bayesian_predicted_risk_percent", ascending=False)
region.to_csv(OUT/"regional_bayesian_risk_summary.csv", index=False)

w = df["weight"].to_numpy(float); y = df["high_risk_unreached"].to_numpy(float); total_hr = (y*w).sum(); total_w = w.sum(); rng = np.random.default_rng(20260704)
reg_score = df.groupby("region").apply(lambda g: wmean(g["high_risk_unreached"], g["weight"])).to_dict()
df["region_burden_score"] = df["region"].map(reg_score).astype(float)
strategies = {"Bayesian risk targeting":"bayesian_risk", "Climate/livelihood shock-only":"climate_livelihood_shock", "Drought-only":"drought_shock", "Poverty-only":"poor", "Region-burden targeting":"region_burden_score", "Direct-needs benchmark":"direct_need_score"}
rows=[]
for name,col in strategies.items():
    for b in [.05,.10,.15,.20,.25,.30,.40,.50]:
        score = df[col].to_numpy(float) + rng.uniform(0, 1e-9, len(df))
        order = np.argsort(-score); selected = (np.cumsum(w[order]) <= b*total_w).astype(float)
        reached = (y[order]*selected*w[order]).sum(); recall = reached/total_hr; missed = 1-recall
        rows.append({"strategy": name, "support_coverage_percent": b*100, "recall_mean": recall*100, "missed_mean": missed*100})
# random average
for b in [.05,.10,.15,.20,.25,.30,.40,.50]:
    vals=[]
    for _ in range(200):
        order=np.argsort(-rng.uniform(size=len(df))); selected=(np.cumsum(w[order]) <= b*total_w).astype(float); vals.append((y[order]*selected*w[order]).sum()/total_hr)
    rows.append({"strategy":"Random targeting", "support_coverage_percent": b*100, "recall_mean": np.mean(vals)*100, "missed_mean": (1-np.mean(vals))*100})
targeting = pd.DataFrame(rows); targeting.to_csv(OUT/"targeting_simulation_results.csv", index=False)

# 7) Main figure
fig, ax = plt.subplots(figsize=(10, 6))
for name in list(strategies.keys()) + ["Random targeting"]:
    g = targeting[targeting.strategy == name]
    ax.plot(g.support_coverage_percent, g.missed_mean, marker="o", linewidth=2.8 if name in ["Bayesian risk targeting","Direct-needs benchmark"] else 1.7, label=name)
ax.set_title("Who Would We Miss?"); ax.set_xlabel("Households reached by support (%)"); ax.set_ylabel("High-risk unreached households missed (%)")
ax.set_ylim(0,100); ax.grid(True, alpha=.25); ax.legend(frameon=False, ncol=2); fig.tight_layout()
fig.savefig(FIG/"figure_1_missed_households_curve.png", dpi=300)
print("Analysis complete. See figures/ and outputs/.")
