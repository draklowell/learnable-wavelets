import copy
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Generator


class Leaf(Enum):
    KEEP = "keep"
    DROP = "drop"

    def encode(self) -> str:
        return self.value


@dataclass
class Node:
    wavelet: str
    hh: "Node | Leaf"
    ll: "Node | Leaf"
    hl: "Node | Leaf"

    def encode(self) -> str:
        return (
            f"({self.wavelet},{self.hh.encode()},{self.ll.encode()},{self.hl.encode()})"
        )


@dataclass
class Tree:
    root: Node
    support_sizes: dict[str, int]

    @staticmethod
    def _build_node_config(node: Node | Leaf) -> dict:
        if isinstance(node, Leaf):
            return node.value

        return {
            "wavelet": node.wavelet,
            "padding": "reflect",
            "hh": Tree._build_node_config(node.hh),
            "hl": Tree._build_node_config(node.hl),
            "lh": Tree._build_node_config(node.hl),
            "ll": Tree._build_node_config(node.ll),
        }

    def build_config(self) -> dict:
        wavelets = []
        for name, support_size in self.support_sizes.items():
            wavelets.append({"name": name, "support_size": support_size})

        return {
            "wavelets": wavelets,
            "tree": self._build_node_config(self.root),
        }

    @staticmethod
    def _iter_nodes(node: Node | Leaf) -> Generator[Node, None, None]:
        if isinstance(node, Leaf):
            return

        yield node
        yield from Tree._iter_nodes(node.hh)
        yield from Tree._iter_nodes(node.hl)
        yield from Tree._iter_nodes(node.ll)

    def simplify(self):
        rename_map = {}

        # Iter nodes is stable
        for node in self._iter_nodes(self.root):
            if node.wavelet not in rename_map:
                new_name = f"w{len(rename_map)}"
                rename_map[node.wavelet] = new_name

        self.support_sizes = {
            rename_map[name]: support_size
            for name, support_size in self.support_sizes.items()
            if name in rename_map  # Ignore wavelets that are not used in the tree
        }

        for node in self._iter_nodes(self.root):
            node.wavelet = rename_map[node.wavelet]

    def get_hash(self, is_simple: bool = False) -> str:
        if not is_simple:
            self = copy.deepcopy(self)

        encoded = self.root.encode()

        for name, support_size in sorted(self.support_sizes.items()):
            encoded += f"{name}:{support_size};"

        return hashlib.sha512(encoded.encode()).hexdigest()
