from .openpose_render_nodes import OpenposeRenderNode


NODE_CLASS_MAPPINGS = {
    "OpenposeRenderNode": OpenposeRenderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposeRenderNode": "Openpose Render Node",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
