"""Hy-Nomas (tencent/Hy-Nomas) as native ComfyUI nodes: a video world model generated segment by segment."""
WEB_DIRECTORY = "./web"  # the camera recorder widget

if __package__:  # ComfyUI imports the pack as a package; pytest may import this file on its own
    from .nodes import comfy_entrypoint

    __all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
