import numpy as np

def aggregate_to_network_fc(fc, roi_to_net, n_networks):
    net_fc = np.zeros((n_networks, n_networks))
    for i in range(n_networks):
        idx_i = np.where(roi_to_net == i)[0]
        for j in range(n_networks):
            idx_j = np.where(roi_to_net == j)[0]
            vals = fc[np.ix_(idx_i, idx_j)]
            net_fc[i, j] = np.nanmean(vals)
    net_fc = (net_fc + net_fc.T) / 2
    #np.fill_diagonal(net_fc, 0)
    return net_fc