import pandas as pd
from sklearn.neighbors import NearestNeighbors

def load_metadata(path: str) -> pd.DataFrame:
    demos = pd.read_csv(path, sep=",", skiprows=1, encoding="latin1")
    meta = demos[["subjectkey", "sex", "interview_age", "Group", "Primary_Dx"]].copy()
    meta["age_years"] = meta["interview_age"] / 12.0

    def assign_group(row):
        if row["Group"] == "GenPop":
            return "HC"
        elif str(row["Primary_Dx"]).upper() == "MDD":
            return "MDD"
        else:
            return "OtherPatient"

    meta["Group_clean"] = meta.apply(assign_group, axis=1)
    meta = meta[meta["Group_clean"].isin(["MDD", "HC"])]
    return meta.drop_duplicates(subset=["subjectkey"])

def match_mdd_hc(meta: pd.DataFrame, mdd_ids: list[str], hc_ids: list[str]) -> pd.DataFrame:
    meta_hc = meta[(meta["Group_clean"] == "HC") & (meta["subjectkey"].isin(hc_ids))].copy()
    meta_mdd = meta[(meta["Group_clean"] == "MDD") & (meta["subjectkey"].isin(mdd_ids))].copy()
    
    matched_pairs = []

    for sex in meta_mdd["sex"].unique():
        mdd_subset = meta_mdd[meta_mdd["sex"] == sex]
        hc_subset = meta_hc[meta_hc["sex"] == sex].copy()
        used_hc = set()

        # If no HC of that sex exists (e.g., sex == "O"), fall back to all available HCs
        if hc_subset.empty:
            print(f"No HC subjects with sex = {sex}; using all HCs as fallback.")
            hc_subset = meta_hc.copy()

        # Build nearest neighbor model on age
        nn = NearestNeighbors(n_neighbors=len(hc_subset))
        nn.fit(hc_subset[["age_years"]])

        for _, row in mdd_subset.iterrows():
            distances, indices = nn.kneighbors([[row["age_years"]]])
            for idx in indices[0]:
                hc_id = hc_subset.iloc[idx]["subjectkey"]
                if hc_id not in used_hc:
                    matched_pairs.append((row["subjectkey"], hc_id))
                    used_hc.add(hc_id)
                    break
            else:
                # If all HCs are used, reuse the closest one (to ensure every MDD gets matched)
                closest_hc_id = hc_subset.iloc[indices[0][0]]["subjectkey"]
                matched_pairs.append((row["subjectkey"], closest_hc_id))
                print(f"Reused HC {closest_hc_id} for MDD {row['subjectkey']} (no unused left).")

    matched_df = pd.DataFrame(matched_pairs, columns=["MDD_subject", "HC_subject"])
    print(f"Matched {len(matched_df)} pairs total")
    return matched_df

