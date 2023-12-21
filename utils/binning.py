import numpy as np

ebin_edges = np.geomspace(0.2, 2000, 41)
ebin_engs = np.sqrt(ebin_edges[1:] * ebin_edges[:-1])
ebin_names = [f'{ebin_edges[i]:.2f}-{ebin_edges[i+1]:.2f} GeV' for i in range(len(ebin_edges)-1)]