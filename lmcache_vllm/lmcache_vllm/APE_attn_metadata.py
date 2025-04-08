from dataclasses import dataclass
import torch
from vllm.attention import AttentionMetadata

@dataclass
class APEMetadata:
    """Metadata for APE (Adaptive Parallel Encoding)"""
    temperature: float  # Temperature parameter for APE
    scale: float  # Scale parameter for APE
    positions: torch.Tensor  # Position indices for APE
    processed_layer_count: int = 0  # Track processed layers

# Extend AttentionMetadata to include APE metadata
def extend_attention_metadata():
    # Add APE metadata to AttentionMetadata
    AttentionMetadata.ape_metadata = None
    
    # Add method to set APE metadata
    def set_ape_metadata(self, temperature, scale, positions):
        self.ape_metadata = APEMetadata(
            temperature=temperature,
            scale=scale,
            positions=positions,
            processed_layer_count=0
        )
    
    # Add the method to AttentionMetadata
    AttentionMetadata.set_ape_metadata = set_ape_metadata 