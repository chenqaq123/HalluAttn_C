"""SinkDetect: Sink-Purified Attention for Hallucination Detection."""

from .adapter import DetectionAdapter, get_llm_layers, inject_detection_adapter
from .sink_utils import auto_detect_sinks, purify_attention, top_mass_mask_visual

__all__ = [
    "DetectionAdapter",
    "get_llm_layers",
    "inject_detection_adapter",
    "auto_detect_sinks",
    "purify_attention",
    "top_mass_mask_visual",
]
