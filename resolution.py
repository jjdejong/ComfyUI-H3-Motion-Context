"""Resolution helpers for MiniMax H3 workflows."""

import math

from comfy_api.latest import io


class MiniMaxH3ResolutionFromImage(io.ComfyNode):
    """Scale an image aspect ratio to a target pixel area on H3's grid."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ResolutionFromImage",
            display_name="MiniMax H3 Resolution From Image",
            category="utilities/minimax",
            description=(
                "Calculate one shared H3 width and height from an image's "
                "aspect ratio and a target megapixel count."),
            inputs=[
                io.Image.Input("image"),
                io.Float.Input(
                    "megapixels", default=0.5, min=0.1, max=16.0, step=0.1,
                    tooltip="Target total megapixels before H3 grid rounding."),
                io.Int.Input(
                    "multiple", default=32, min=32, max=128, step=32,
                    advanced=True,
                    tooltip="Round both dimensions to this H3 canvas multiple."),
            ],
            outputs=[
                io.Int.Output("width"),
                io.Int.Output("height"),
            ],
        )

    @classmethod
    def execute(cls, image, megapixels, multiple):
        height, width = image.shape[1:3]
        target_pixels = float(megapixels) * 1024 * 1024
        scale = math.sqrt(target_pixels / (width * height))
        width = max(multiple, round(width * scale / multiple) * multiple)
        height = max(multiple, round(height * scale / multiple) * multiple)
        return io.NodeOutput(width, height)
