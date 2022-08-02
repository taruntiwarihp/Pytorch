# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained(
#             'bert-base-uncased', do_lower=True)


# sample = 'where is Himalayas in the world map?'
# encoding = tokenizer.encode(sample, return_attention_mask=True)
# print(encoding)
# print(tokenizer.convert_ids_to_tokens(encoding))


# sample = 'where is Himalayasss in the world map?'
# encoding = tokenizer.encode(sample)
# print(encoding)
# print(tokenizer.convert_ids_to_tokens(encoding))



# t = tokenizer.encode_plus(
#              sample, max_length=40,
#              pad_to_max_length=True, return_attention_mask=True, 
#              return_token_type_ids=False, truncation=True
#         )

# print(t)

from models import utils, caption
import torch, os
from utils.config import Config
from utils.log_utils import create_logger, get_model_summary
import pprint

def check_dir(opts):

    if not os.path.isdir(opts.dir):
        raise ValueError('Data Dir shouldnot be empty')

    os.makedirs(opts.logDir, exist_ok=True)
    os.makedirs(opts.modelDir, exist_ok=True)


opts = Config()
check_dir(opts)
device = torch.device(opts.device)
model, criterion = caption.build_model(opts)
# model.to(device)


# n_parameters = sum(p.numel()
#                 for p in model.parameters() if p.requires_grad)
# print(f"Number of params: {n_parameters}")

# print(model.named_parameters())


# print(f"Number of params: {n_parameters}")
# logger, tb_log_dir = create_logger(opts)
# logger.info(pprint.pformat(opts))
# logger.info(pprint.pformat(model))
# logger.info(pprint.pformat("Number of params: {n_parameters}"))
# param_dicts = [
# {"params": [p for n, p in model.named_parameters() 
# if "backbone" not in n and p.requires_grad]},
# {
#         "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
#         "lr": opts.lr_backbone,
# },
# ]

# print(param_dicts)
# for name, param in model.named_parameters():
#     print(name)


import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch import optim
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef

t1 = torch.rand((3, 3), requires_grad=True)
t2 = torch.rand((3, 3), requires_grad=True)
rref = rpc.remote("worker1", torch.add, args=(t1, t2))
ddp_model = DDP(model)

# Setup optimizer
optimizer_params = [rref]
for param in ddp_model.parameters():
    optimizer_params.append(RRef(param))

dist_optim = DistributedOptimizer(
    optim.SGD,
    optimizer_params,
    lr=0.05,
)

with dist_autograd.context() as context_id:
    pred = ddp_model(rref.to_here())
    # loss = criterion(pred, target)
    # dist_autograd.backward(context_id, [loss])
    # dist_optim.step(context_id)


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def example(rank, world_size):
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model

    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()