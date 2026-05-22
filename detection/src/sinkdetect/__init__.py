"""SinkDetect detection source package."""

__all__ = [
    "DetectionAdapter",
    "get_llm_layers",
    "inject_detection_adapter",
    "auto_detect_sinks",
    "purify_attention",
    "top_mass_mask_visual",
]


def __getattr__(name):
    if name in {"DetectionAdapter", "get_llm_layers", "inject_detection_adapter"}:
        from . import adapter

        return getattr(adapter, name)
    if name in {"auto_detect_sinks", "purify_attention", "top_mass_mask_visual"}:
        from . import sink_utils

        return getattr(sink_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
