import copy
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


EPS = 1e-8


class Node:
    def __init__(self, kind: str, value=None, children: Optional[List["Node"]] = None):
        self.kind = kind
        self.value = value
        self.children = children or []

    def clone(self) -> "Node":
        return copy.deepcopy(self)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "var":
            return x[:, self.value]
        if self.kind == "const":
            return np.full(x.shape[0], self.value, dtype=float)

        vals = [c.evaluate(x) for c in self.children]
        op = self.value

        if op == "add":
            out = vals[0] + vals[1]
        elif op == "sub":
            out = vals[0] - vals[1]
        elif op == "mul":
            out = vals[0] * vals[1]
        elif op == "pdiv":
            den = np.where(np.abs(vals[1]) < EPS, EPS, vals[1])
            out = vals[0] / den
        elif op == "sin":
            out = np.sin(vals[0])
        elif op == "cos":
            out = np.cos(vals[0])
        elif op == "log":
            out = np.log(np.abs(vals[0]) + EPS)
        elif op == "exp":
            out = np.exp(np.clip(vals[0], -20.0, 20.0))
        else:
            raise ValueError(f"Unsupported op: {op}")

        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def to_string(self) -> str:
        if self.kind == "var":
            return f"x{self.value}"
        if self.kind == "const":
            return f"{self.value:.4f}"

        op = self.value
        if op in {"add", "sub", "mul", "pdiv"}:
            left = self.children[0].to_string()
            right = self.children[1].to_string()
            symbols = {"add": "+", "sub": "-", "mul": "*", "pdiv": "/"}
            return f"({left} {symbols[op]} {right})"
        return f"{op}({self.children[0].to_string()})"

    def structure_signature(self) -> str:
        if self.kind == "var":
            return "T"
        if self.kind == "const":
            return "T"
        child_signatures = ",".join(c.structure_signature() for c in self.children)
        return f"{self.value}({child_signatures})"


@dataclass
class Individual:
    tree: Node
    fitness: float = float("inf")


FUNCTION_SET = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "pdiv": 2,
    "sin": 1,
    "cos": 1,
    "log": 1,
    "exp": 1,
}


def generate_terminal(num_features: int, rng: random.Random, const_min: float, const_max: float) -> Node:
    if rng.random() < 0.7:
        return Node("var", value=rng.randrange(num_features))
    return Node("const", value=rng.uniform(const_min, const_max))


def generate_random_tree(
    num_features: int,
    max_depth: int,
    rng: random.Random,
    const_min: float,
    const_max: float,
    method: str,
) -> Node:
    if max_depth <= 1:
        return generate_terminal(num_features, rng, const_min, const_max)

    if method == "grow" and rng.random() < 0.35:
        return generate_terminal(num_features, rng, const_min, const_max)

    op = rng.choice(list(FUNCTION_SET.keys()))
    arity = FUNCTION_SET[op]
    children = [
        generate_random_tree(num_features, max_depth - 1, rng, const_min, const_max, method)
        for _ in range(arity)
    ]
    return Node("func", value=op, children=children)


def collect_nodes_with_parents(
    node: Node,
    parent: Optional[Node] = None,
    child_idx: Optional[int] = None,
    out: Optional[List[Tuple[Node, Optional[Node], Optional[int]]]] = None,
) -> List[Tuple[Node, Optional[Node], Optional[int]]]:
    if out is None:
        out = []
    out.append((node, parent, child_idx))
    for i, c in enumerate(node.children):
        collect_nodes_with_parents(c, node, i, out)
    return out
