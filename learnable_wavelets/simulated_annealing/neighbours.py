import copy
import random
from itertools import combinations
from typing import Iterator

from learnable_wavelets.simulated_annealing.moves import (
    Context,
    Move,
    ReplaceLeaf,
    ReplaceLeafWithNode,
    ReplaceNodeWithLeaf,
    ReplaceWavelet,
    SwapLeaves,
    get_score,
)
from learnable_wavelets.simulated_annealing.tree import Leaf, Node, Tree


def get_tree_depth(node: Node | Leaf) -> int:
    if isinstance(node, Leaf):
        return 0

    return 1 + max(
        get_tree_depth(node.hh),
        get_tree_depth(node.ll),
        get_tree_depth(node.hl),
    )


def iter_leaves(
    node: Node | Leaf, path: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    if isinstance(node, Leaf):
        yield path
        return

    yield from iter_leaves(
        node.hh,
        path + ("hh",),
    )
    yield from iter_leaves(
        node.ll,
        path + ("ll",),
    )
    yield from iter_leaves(
        node.hl,
        path + ("hl",),
    )


def iter_pre_leaves(
    node: Node | Leaf, path: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    if isinstance(node, Leaf):
        return

    if (
        isinstance(node.hh, Leaf)
        and isinstance(node.ll, Leaf)
        and isinstance(node.hl, Leaf)
    ):
        yield path
        return

    yield from iter_pre_leaves(
        node.hh,
        path + ("hh",),
    )
    yield from iter_pre_leaves(
        node.ll,
        path + ("ll",),
    )
    yield from iter_pre_leaves(
        node.hl,
        path + ("hl",),
    )


def iter_nodes(
    node: Node | Leaf, path: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    if isinstance(node, Leaf):
        return

    yield path
    yield from iter_nodes(
        node.hh,
        path + ("hh",),
    )
    yield from iter_nodes(
        node.ll,
        path + ("ll",),
    )
    yield from iter_nodes(
        node.hl,
        path + ("hl",),
    )


def random_neighbour(
    tree: Tree,
    max_score: float,
    max_depth: int = 10,
    new_wavelet_prob: float = 0.15,
    support_sizes: list[int] = [2, 4, 6, 8, 10, 12, 14],
    rng: random.Random = None,
) -> Tree:
    if rng is None:
        rng = random.Random()

    new_tree = copy.deepcopy(tree)
    current_score = get_score(new_tree.root)
    ctx = Context(
        tree=new_tree,
        score=current_score,
        max_score=max_score,
        new_wavelet_prob=new_wavelet_prob,
        max_depth=max_depth,
        support_sizes=support_sizes,
        rng=rng,
    )

    feasible_moves: list[Move] = []

    leaves = list(iter_leaves(ctx.tree.root))
    for path in leaves:
        if ReplaceLeaf.can_apply(path, ctx):
            feasible_moves.append(ReplaceLeaf(path))

        if ReplaceLeafWithNode.can_apply(path, ctx):
            feasible_moves.append(ReplaceLeafWithNode(path))

    for path1, path2 in combinations(leaves, 2):
        if SwapLeaves.can_apply(path1, path2, ctx):
            feasible_moves.append(SwapLeaves(path1, path2))

    for path in iter_pre_leaves(ctx.tree.root):
        if not path:
            continue
        feasible_moves.append(ReplaceNodeWithLeaf(path))

    for path in iter_nodes(ctx.tree.root):
        feasible_moves.append(ReplaceWavelet(path))

    if not feasible_moves:
        raise ValueError("No feasible moves found")

    move = rng.choice(feasible_moves)

    move.apply(ctx)

    new_score = get_score(ctx.tree.root)
    assert (
        new_score <= max_score
    ), f"Move {move} resulted in score {new_score} which is greater than max_score {max_score}"
    new_depth = get_tree_depth(ctx.tree.root)
    assert (
        new_depth <= max_depth
    ), f"Move {move} resulted in depth {new_depth} which is greater than max_depth {max_depth}"

    return new_tree
