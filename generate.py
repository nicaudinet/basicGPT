import torch
import argparse
from transformer import *

name = "Generate Shakespeare"
description = "Load ai-shakespeare and get it to write you a play"
parser = argparse.ArgumentParser(prog=name, description=description)
parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Destination filepath"
    )
parser.add_argument(
        "-n",
        "--num_chars",
        default=500,
        type=int,
        help="Number of characters to generate"
    )
parser.add_argument(
        "-c",
        "--context",
        default=None,
        type=str,
        help="The context for the AI",
    )
args = parser.parse_args()

model = torch.load("ai-shakespeare")
m = model.to(device)

if args.context is not None:
    context = encode(args.context)
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context[None, :]
else:
    context = torch.zeros((1,1), dtype=torch.long, device=device)

print(context)

out = decode(m.generate(context, max_new_tokens=args.num_chars)[0].tolist())

if args.output is not None:
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(out)
else:
    print(out)
