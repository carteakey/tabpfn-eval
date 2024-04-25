from tabpfn.scripts.model_builder import get_model
import matplotlib.pyplot as plt
from tabpfn.priors.utils import plot_prior, plot_features
import numpy as np
import sys
import os

base_dir = '/Users/kchauhan/repos/mds-tmu-dl/'
sys.path.append(os.path.join(base_dir))
sys.path.append(os.path.join(base_dir, 'src/models/tabpfn/modified/'))
from src.models.tabpfn.modified.train_config import config_sample

config_sample['batch_size'] = 4
model = get_model(config_sample, 'cpu', should_train=False,
                  verbose=2)  # , state_dict=model[2].state_dict()

(hp_embedding, data, _), targets, single_eval_pos = next(iter(model[3]))

fig = plt.figure(figsize=(8, 8))
N = 100
plot_features(data[0:N, 0, 0:4], targets[0:N, 0], fig=fig)
# Save the figure
plt.savefig('img/prior_features.png')

d = np.concatenate([data[:, 0, :].T, np.expand_dims(targets[:, 0], -1).T])
d[np.isnan(d)] = 0
c = np.corrcoef(d)
plt.matshow(np.abs(c), vmin=0, vmax=1)
plt.show()
