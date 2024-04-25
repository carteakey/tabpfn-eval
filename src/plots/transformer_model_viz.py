import sys
import os
from torchviz import make_dot
from torchview import draw_graph
from tabpfn.scripts.model_builder import load_model_only_inference, get_model
from src.models.tabpfn.modified.train_config import config_sample

base_dir = '/home/kchauhan/repos/mds-tmu-dl'
sys.path.append(os.path.join(base_dir))
sys.path.append(os.path.join(base_dir, 'src/models/tabpfn/modified/'))


# Load the model for inference
model_tuple, config = load_model_only_inference(
    os.path.join(base_dir, "src/models/tabpfn/modified/models_diff/"),
    "prior_diff_real_checkpointkc_n_0_epoch_100.cpkt", "cpu"
)
model = model_tuple[2]

print(model)

# Set batch size in the config
config_sample['batch_size'] = 4

# Get the model for evaluation
model_2 = get_model(config_sample, 'cpu', should_train=False, verbose=2)
input, targets, single_eval_pos = next(iter(model_2[3]))

# Perform forward pass
output = model(input, single_eval_pos=single_eval_pos)

# Visualize the model using torchviz
dot = make_dot(
    output,
    params=dict(model.named_parameters())
)
dot.format = 'png'
dot.render('img/transformer_model_viz')

# Visualize the model using torchview
model_graph = draw_graph(model,
                         input_data=(input, None, single_eval_pos),
                         hide_module_functions=False).visual_graph

model_graph.render(directory='img',cleanup=True)
