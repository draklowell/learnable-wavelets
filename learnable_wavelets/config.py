from enum import Enum

import yaml
from pydantic import BaseModel, Field


class LeafNode(str, Enum):
    KEEP = "keep"
    DROP = "drop"


class PaddingMode(str, Enum):
    REFLECT = "reflect"
    CONSTANT = "constant"
    CIRCULAR = "circular"
    REPLICATE = "replicate"


class SplitNode(BaseModel):
    wavelet: str = Field(
        description="Name of the wavelet to use for this node (if not a leaf)"
    )
    padding: PaddingMode = Field(
        description="Padding mode to use for this node (if not a leaf)"
    )
    hh: "SplitNode | LeafNode" = Field(description="Configuration for the HH subband")
    hl: "SplitNode | LeafNode" = Field(description="Configuration for the HL subband")
    lh: "SplitNode | LeafNode" = Field(description="Configuration for the LH subband")
    ll: "SplitNode | LeafNode" = Field(description="Configuration for the LL subband")


SplitNode.model_rebuild()


class WaveletDefinition(BaseModel):
    name: str = Field(description="Name of the wavelet (e.g., 'haar', 'sym2')")
    support_size: int = Field(
        description="Support size of the wavelet filter", gt=0, multiple_of=2
    )


class ModuleConfig(BaseModel):
    wavelets: list[WaveletDefinition] = Field(description="List of wavelet definitions")
    tree: SplitNode = Field(
        description="Hierarchical tree structure for the wavelet transform"
    )

    def count_compression_rate(self, total_width, total_height) -> float:
        def count_node(node: SplitNode | LeafNode, width, height) -> int:
            if node == LeafNode.KEEP:
                return width * height
            elif node == LeafNode.DROP:
                return 0

            for wavelet in self.wavelets:
                if wavelet.name == node.wavelet:
                    K = wavelet.support_size
                    break
            else:
                raise ValueError("Broken tree")

            return (
                count_node(node.hh, (width + K - 1) // 2, (height + K - 1) // 2)
                + count_node(node.hl, (width + K - 1) // 2, (height + K - 1) // 2)
                + count_node(node.lh, (width + K - 1) // 2, (height + K - 1) // 2)
                + count_node(node.ll, (width + K - 1) // 2, (height + K - 1) // 2)
            )

        total_size = total_width * total_height
        retained_size = count_node(self.tree, total_width, total_height)
        return retained_size / total_size


def load_config(path: str) -> ModuleConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return ModuleConfig.model_validate(raw)
