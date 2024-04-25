import torch
from torchviz import make_dot
from tabpfn.scripts.model_builder import load_model_only_inference, get_model
import sys
import os

base_dir = '/Users/kchauhan/repos/mds-tmu-dl/'
sys.path.append(os.path.join(base_dir))
sys.path.append(os.path.join(base_dir, 'src/models/tabpfn/modified/'))
from src.models.tabpfn.modified.train_config import config_sample

model_tuple, config = load_model_only_inference(
    os.path.join(base_dir, "src/models/tabpfn/modified/models_diff/"),
    "prior_diff_real_checkpointkc_n_0_epoch_100.cpkt", "cuda")
model = model_tuple[2]

print(model)

config_sample['batch_size'] = 4
model_2 = get_model(config_sample, 'cpu', should_train=False,
                    verbose=2)  # , state_dict=model[2].state_dict()
input, targets, single_eval_pos = next(iter(model_2[3]))

output = model(input, single_eval_pos=single_eval_pos)

# now we can visualize this model
dot = make_dot(
    output,
    params=dict(model.named_parameters()),
)
dot.format = 'png'
dot.render('img/transformer_model_viz')
