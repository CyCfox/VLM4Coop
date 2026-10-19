"""Headless global LBF RGB renderer with visual ID/level/coordinate annotations.

Renders only the current field and player positions/levels; no plans, actions,
reward, observation vectors or hidden state are sent to the teacher.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np

RENDERER_VERSION = "pil-global-grid-v3-zero-based-food-ids"


def global_rgb(env, food_ids):
    cell, margin = 56, 28
    rows, cols = env.field.shape
    image = Image.new("RGB", (cols * cell + margin, rows * cell + margin), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    for row in range(rows):
        draw.text((2, margin + row * cell + 18), str(row), fill="black", font=font)
    for col in range(cols):
        draw.text((margin + col * cell + 20, 2), str(col), fill="black", font=font)
    for row in range(rows):
        for col in range(cols):
            x, y = margin + col * cell, margin + row * cell
            draw.rectangle((x, y, x + cell, y + cell), outline="#a0a0a0", width=1)
            level = int(env.field[row, col])
            if level:
                draw.rectangle((x + 5, y + 5, x + cell - 5, y + cell - 5), fill="#9be49b", outline="#246b24", width=2)
                draw.text((x + 10, y + 8), f"F{food_ids[(row, col)]}", fill="black", font=font)
                draw.text((x + 10, y + 28), f"L{level}", fill="black", font=font)
    for agent_id, player in enumerate(env.players):
        row, col = player.position
        x, y = margin + col * cell, margin + row * cell
        draw.ellipse((x + 3, y + 3, x + cell - 3, y + cell - 3), fill="#b6d8fa", outline="#174c85", width=2)
        draw.text((x + 10, y + 7), f"A{agent_id}", fill="black", font=font)
        draw.text((x + 10, y + 29), f"L{player.level}", fill="black", font=font)
    return np.asarray(image)
