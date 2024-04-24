import torch
from torchviz import make_dot
from tabpfn.scripts.model_builder import load_model_only_inference
from tabpfn.priors.fast_gp import DataLoader
from tabpfn.utils import get_uniform_single_eval_pos_sampler
from torchsummary import summary

model_tuple, config = load_model_only_inference(
    "/home/kchauhan/repos/mds-tmu-dl/src/models/tabpfn/modified/models_diff/",
    "prior_diff_real_checkpointkc_n_42_epoch_46.cpkt", "cuda")
model = model_tuple[2]

print(model)


def eval_pos_seq_len_sampler():
    single_eval_pos = get_uniform_single_eval_pos_sampler(1024, 0)
    return single_eval_pos, 1024

dl = DataLoader(num_steps=1,
                num_features=100,
                batch_size=1,
                device='cuda',
                config=config,
                eval_pos_seq_len_sampler=eval_pos_seq_len_sampler)
dl.model = model

for batch, (data, targets, single_eval_pos) in enumerate(dl):
    single_eval_pos =single_eval_pos() if callable(single_eval_pos) else single_eval_pos
    input = tuple(
        e.to('cuda') if torch.is_tensor(e) else e
        for e in data) if isinstance(data, tuple) else data.to('cuda')
    output = model(input,
                   single_eval_pos=single_eval_pos)
    
    # now we can visualize this model
    dot = make_dot(output, params=dict(model.named_parameters()),)
    dot.format = 'png'
    dot.render('img/transformer_model_viz')
