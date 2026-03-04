"""
utils/io/io_atlas.py
--------------------
Load brain atlas ROI-to-network mapping from CSV.
"""

import numpy as np
import pandas as pd


def load_atlas(atlas_path):
    """
    Load ROI-to-network mapping and network name list from a CSV file.

    Expected columns: NetworkID (int), NetworkName (str)

    Returns
    -------
    roi_to_net : (R,) int array  – network index for each ROI
    n_networks : int
    net_names  : list[str]       – ordered list of network names
    """
    df = pd.read_csv(atlas_path)
    df["NetworkID"]   = pd.to_numeric(df["NetworkID"], errors="coerce").astype(int)
    df["NetworkName"] = df["NetworkName"].astype(str)
    roi_to_net = df["NetworkID"].values
    n_networks = len(np.unique(roi_to_net))
    net_names  = df.groupby("NetworkID")["NetworkName"].first().to_list()
    return roi_to_net, n_networks, net_names
