"""Augmentation nur für den Trainings-Split. Wird nicht in die Dateien geschrieben."""

from PIL import Image, ImageEnhance, ImageOps

BACKGROUND = (255, 255, 255)

POLICY = {
    "applies_to": "train",
    "horizontal_flip": 0.5,
    "rotation_degrees": 15,
    "brightness_factor": (0.8, 1.2)
}


def preview_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
    rgb = image.convert("RGB")
    low, high = POLICY["brightness_factor"]
    rotated = rgb.rotate(POLICY["rotation_degrees"], expand=False, fillcolor=BACKGROUND)
    return [
        ("original", rgb),
        ("flip", ImageOps.mirror(rgb)),
        (f"rotate {POLICY['rotation_degrees']}°", rotated),
        (f"brightness {low}", ImageEnhance.Brightness(rgb).enhance(low)),
        (f"brightness {high}", ImageEnhance.Brightness(rgb).enhance(high)),
    ]
