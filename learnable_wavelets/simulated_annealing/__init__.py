import math
import random
from typing import Callable

from learnable_wavelets.config import LeafNode, ModuleConfig, SplitNode
from learnable_wavelets.simulated_annealing.neighbours import random_neighbour
from learnable_wavelets.simulated_annealing.tree import Leaf, Node, Tree


def config_to_tree(config: ModuleConfig) -> "Tree":
    support_sizes = {wavelet.name: wavelet.support_size for wavelet in config.wavelets}

    def build_node(node_config: SplitNode | LeafNode) -> Node | Leaf:
        if node_config == LeafNode.KEEP:
            return Leaf(Leaf.KEEP)
        elif node_config == LeafNode.DROP:
            return Leaf(Leaf.DROP)

        return Node(
            wavelet=node_config.wavelet,
            hh=build_node(node_config.hh),
            ll=build_node(node_config.ll),
            hl=build_node(node_config.hl),
        )

    tree = Tree(root=build_node(config.tree), support_sizes=support_sizes)
    return tree


class SimulatedAnnealing:
    def __init__(
        self,
        initial_config: ModuleConfig,
        objective: Callable[[list[tuple[dict, str]]], list[float]],
        max_score: float,
        max_depth: int = 10,
        new_wavelet_prob: float = 0.15,
        support_sizes: list[int] | None = None,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.01,
        cooling_rate: float = 0.95,
        batches_per_temperature: int = 100,
        batch_size: int = 4,
        rng: random.Random | None = None,
        on_batch_complete: Callable[[dict], None] | None = None,
    ):
        self.initial_tree = config_to_tree(initial_config)
        self.initial_tree.simplify()

        self.objective = objective

        self.max_score = max_score
        self.max_depth = max_depth
        self.new_wavelet_prob = new_wavelet_prob
        self.support_sizes = support_sizes or [2, 4, 6, 8, 10, 12, 14]
        self.temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.batches_per_temperature = batches_per_temperature
        self.batch_size = batch_size
        self.on_batch_complete = on_batch_complete

        self.rng = rng or random.Random()

    def _temperature_schedule(self):
        temperature = self.temperature
        while temperature > self.final_temperature:
            yield temperature
            temperature *= self.cooling_rate

    def _acceptance_probability(
        self, current_energy: float, new_energy: float, temperature: float
    ) -> float:
        if new_energy < current_energy:
            return 1.0

        return math.exp((current_energy - new_energy) / temperature)

    def run(self) -> tuple[dict, float]:
        tree = self.initial_tree
        energy = self.objective([(tree.build_config(), tree.get_hash())])[0]

        best_tree = self.initial_tree.copy()
        best_energy = energy

        for temperature in self._temperature_schedule():
            for _ in range(self.batches_per_temperature):
                neighbours = [
                    random_neighbour(
                        tree,
                        self.max_score,
                        self.max_depth,
                        self.new_wavelet_prob,
                        self.support_sizes,
                        self.rng,
                    )
                    for _ in range(self.batch_size)
                ]
                for neighbour in neighbours:
                    neighbour.simplify()
                energies = self.objective(
                    [
                        (neighbour.build_config(), neighbour.get_hash())
                        for neighbour in neighbours
                    ]
                )

                if self.on_batch_complete is not None:
                    self.on_batch_complete(
                        {
                            "temperature": temperature,
                            "batch_size": len(neighbours),
                            "current_energy": energy,
                            "best_energy": best_energy,
                        }
                    )

                for neighbour, neighbour_energy in zip(neighbours, energies):
                    if self.rng.random() < self._acceptance_probability(
                        energy, neighbour_energy, temperature
                    ):
                        tree = neighbour
                        energy = neighbour_energy

                    if energy < best_energy:
                        best_tree = tree.copy()
                        best_energy = energy

        return best_tree.build_config(), best_energy


__all__ = (
    "SimulatedAnnealing",
    "config_to_tree",
)
