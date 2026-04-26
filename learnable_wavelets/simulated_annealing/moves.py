import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from learnable_wavelets.simulated_annealing.tree import Leaf, Node, Tree


def get_score(tree: Node | Leaf) -> float:
    if isinstance(tree, Leaf):
        return 1.0 if tree == Leaf.KEEP else 0.0

    return (
        0.25 * get_score(tree.hh) + 0.25 * get_score(tree.ll) + 0.5 * get_score(tree.hl)
    )


# Support moves are (single at a time):
# 1. Replace leaf KEEP <-> DROP
# 2. Swap two leaves KEEP <-> DROP (to move on the constraint boundary)
# 3. Replace a leaf with a node (wavelet - random existing uniformly or new with fixed probability)
#    - the children types are sampled with p=2/3 (same type) and p=1/3 (different type) and to satisfy
#    - the score constraint
# 4. Replace a node with a leaf (KEEP/DROP is chosen with probability proportional to the number of
#    KEEP/DROP leaves and such that the score constraint is satisfied)
# 5. Replace a wavelet in a node with another wavelet (random existing uniformly or new with fixed probability)


@dataclass
class Context:
    tree: Tree
    score: float
    max_score: float
    new_wavelet_prob: float
    # Max number of nodes in depth
    max_depth: int
    support_sizes: list[int]
    rng: random.Random

    def generate_support_size(self) -> int:
        return self.rng.choice(self.support_sizes)

    def generate_wavelet(self, old_wavelet: str | None = None) -> str:
        if old_wavelet is not None:
            possible_wavelets = set(self.tree.support_sizes.keys()) - {old_wavelet}
        else:
            possible_wavelets = set(self.tree.support_sizes.keys())

        if len(possible_wavelets) == 0 or self.rng.random() < self.new_wavelet_prob:
            i = 0
            while f"w{i}" in self.tree.support_sizes:
                i += 1

            self.tree.support_sizes[f"w{i}"] = self.generate_support_size()
            return f"w{i}"

        return self.rng.choice(list(possible_wavelets))

    def get_node_and_parent_at(
        self, path: tuple[str, ...]
    ) -> tuple[Node | Leaf, Node | None, str | None]:
        if not path:
            return self.tree.root, None, None

        parent = None
        node = self.tree.root
        for attr in path:
            if not isinstance(node, Node):
                raise ValueError(f"Path {path} is invalid for the given tree.")
            parent = node
            node = getattr(node, attr)

        return node, parent, path[-1]

    def get_score_at(self, path: tuple[str, ...]) -> float:
        score = 1.0
        for attr in path:
            if attr == "hh":
                score *= 0.25
            elif attr == "ll":
                score *= 0.25
            elif attr == "hl":
                score *= 0.5
            else:
                raise ValueError(f"Invalid attribute {attr} in path {path}.")

        return score


class Move(ABC):
    @abstractmethod
    def apply(self, ctx: Context):
        pass


class ReplaceLeaf(Move):
    def __init__(self, path: tuple[str, ...]):
        self.path = path

    @staticmethod
    def can_apply(path: tuple[str, ...], ctx: Context) -> bool:
        old_value, *_ = ctx.get_node_and_parent_at(path)
        if old_value == Leaf.KEEP:
            return True

        score_path = ctx.get_score_at(path)

        return ctx.score + score_path <= ctx.max_score

    def apply(self, ctx: Context):
        old_value, parent, attr = ctx.get_node_and_parent_at(self.path)
        if parent is None:
            raise ValueError("Cannot replace the root leaf.")

        if not isinstance(old_value, Leaf):
            raise ValueError("The target for ReplaceLeaf must be a Leaf.")

        new_value = Leaf.KEEP if old_value == Leaf.DROP else Leaf.DROP
        setattr(parent, attr, new_value)


class ReplaceNodeWithLeaf(Move):
    def __init__(self, path: tuple[str, ...]):
        self.path = path

    def apply(self, ctx: Context):
        node, parent, attr = ctx.get_node_and_parent_at(self.path)
        if parent is None:
            raise ValueError("Cannot replace the root node.")

        if not isinstance(node, Node):
            raise ValueError("The target for ReplaceNodeWithLeaf must be a Node.")

        score_path = ctx.get_score_at(self.path)
        score = ctx.score - score_path * get_score(node) + score_path

        if score > ctx.max_score:
            setattr(parent, attr, Leaf.DROP)
            return

        prob_keep = (
            (node.hh == Leaf.KEEP) * 0.25
            + (node.ll == Leaf.KEEP) * 0.25
            + (node.hl == Leaf.KEEP) * 0.5
        )

        if ctx.rng.random() < prob_keep:
            new_value = Leaf.KEEP
        else:
            new_value = Leaf.DROP

        setattr(parent, attr, new_value)


class SwapLeaves(Move):
    def __init__(self, path1: tuple[str, ...], path2: tuple[str, ...]):
        self.path1 = path1
        self.path2 = path2

    @staticmethod
    def can_apply(path1: tuple[str, ...], path2: tuple[str, ...], ctx: Context) -> bool:
        if path1 == path2:
            return False

        leaf1, *_ = ctx.get_node_and_parent_at(path1)
        leaf2, *_ = ctx.get_node_and_parent_at(path2)
        score1 = ctx.get_score_at(path1)
        score2 = ctx.get_score_at(path2)

        score = ctx.score
        score -= score1 * (leaf1 == Leaf.KEEP) + score2 * (leaf2 == Leaf.KEEP)
        score += score1 * (leaf2 == Leaf.KEEP) + score2 * (leaf1 == Leaf.KEEP)
        return score <= ctx.max_score

    def apply(self, ctx: Context):
        node1, parent1, attr1 = ctx.get_node_and_parent_at(self.path1)
        node2, parent2, attr2 = ctx.get_node_and_parent_at(self.path2)

        if parent1 is None or parent2 is None:
            raise ValueError("Cannot swap the root leaf.")

        setattr(parent1, attr1, node2)
        setattr(parent2, attr2, node1)


class ReplaceLeafWithNode(Move):
    def __init__(self, path: tuple[str, ...]):
        self.path = path

    @staticmethod
    def can_apply(path: tuple[str, ...], ctx: Context) -> bool:
        return len(path) < ctx.max_depth

    def apply(self, ctx: Context):
        node, parent, attr = ctx.get_node_and_parent_at(self.path)
        if parent is None:
            raise ValueError("Cannot replace the root leaf.")

        if not isinstance(node, Leaf):
            raise ValueError("The target for ReplaceLeafWithNode must be a Leaf.")

        if node == Leaf.KEEP:
            max_children = 4
        else:
            score_per_child = ctx.get_score_at(self.path) * 0.25
            # score + score_per_children * max_children <= max_score
            # score_per_children * max_children <= max_score - score
            # max_children <= (max_score - score) / score_per_child
            max_children = int((ctx.max_score - ctx.score) // score_per_child)

        hh = Leaf.DROP
        ll = Leaf.DROP
        hl = Leaf.DROP
        if max_children < 1:
            pass
        elif max_children < 2:
            # [hh]; [ll]
            rand = ctx.rng.random()
            if rand < 0.5:
                hh = Leaf.KEEP
            else:
                ll = Leaf.KEEP
        elif max_children < 3:
            # [hh, ll]; [hl];
            # [hh]; [ll]
            rand = ctx.rng.random()
            if rand < 0.25:
                hh = Leaf.KEEP
                ll = Leaf.KEEP
            elif rand < 0.5:
                hl = Leaf.KEEP
            elif rand < 0.75:
                hh = Leaf.KEEP
            else:
                ll = Leaf.KEEP
        elif max_children < 4:
            # [hh, ll]; [hl];
            # [hh]; [ll]; [hl, hh]
            # [hl, ll]
            rand = ctx.rng.random()
            if rand < 1 / 6:
                hh = Leaf.KEEP
                ll = Leaf.KEEP
            elif rand < 2 / 6:
                hl = Leaf.KEEP
            elif rand < 3 / 6:
                hh = Leaf.KEEP
            elif rand < 4 / 6:
                ll = Leaf.KEEP
            elif rand < 5 / 6:
                hl = Leaf.KEEP
                hh = Leaf.KEEP
            else:
                hl = Leaf.KEEP
                ll = Leaf.KEEP
        else:
            ll = Leaf.KEEP if ctx.rng.random() < 0.5 else Leaf.DROP
            hh = Leaf.KEEP if ctx.rng.random() < 0.5 else Leaf.DROP
            hl = Leaf.KEEP if ctx.rng.random() < 0.5 else Leaf.DROP

        new_node = Node(ctx.generate_wavelet(), hh, ll, hl)
        setattr(parent, attr, new_node)


class ReplaceWavelet(Move):
    def __init__(self, path: tuple[str, ...]):
        self.path = path

    def apply(self, ctx: Context):
        node, *_ = ctx.get_node_and_parent_at(self.path)

        if not isinstance(node, Node):
            raise ValueError("The target for ReplaceWavelet must be a Node.")

        node.wavelet = ctx.generate_wavelet(node.wavelet)
