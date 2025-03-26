from ursina import *
import numpy as np

app = Ursina()

# Fix window size warning by setting integer values
window.size = (2764, 4915)  # Adjusted to nearest integers

# Define textures
building_texture = load_texture('brick.jpg')
path_texture = load_texture('concrete.png')
tree_texture = load_texture('tree.png')  # Simple tree texture
tree_model = load_model('cube')  # Fallback to cube if custom model missing

# Campus layout coordinates
BUILDINGS = {
    "Geisel Library": {"pos": (0, 0, 0), "scale": (4, 8, 4)},
    "La Jolla House": {"pos": (-10, 0, 5), "scale": (3, 5, 3)},
    "Louis Torres Hall": {"pos": (8, 0, -6), "scale": (4, 6, 4)},
}

PATHS = [
    {"start": (-10, 0, 5), "end": (0, 0, 0), "width": 1},
]

# Create buildings
for name, data in BUILDINGS.items():
    Entity(
        model='cube', texture=building_texture,
        position=data["pos"], scale=data["scale"],
        collider='box', name=name
    )

# Create pathways
for path in PATHS:
    path_length = np.linalg.norm(np.subtract(path["end"], path["start"]))
    Entity(
        model='cube', texture=path_texture,
        position=((path["start"][0] + path["end"][0])/2, 0, (path["start"][2] + path["end"][2])/2),
        scale=(path["width"], 0.1, path_length),
        rotation=(0, np.degrees(np.arctan2(path["end"][0]-path["start"][0], path["end"][2]-path["start"][2])), 0)
    )

# Add terrain
Entity(model='plane', texture='grass', scale=(100, 1, 100), collider='box')

# Create trees using cube model with green color
for _ in range(50):
    Entity(
        model=tree_model,
        texture=tree_texture,
        color=color.green,
        position=(np.random.uniform(-40,40), 0, np.random.uniform(-40,40)),
        scale=(0.5, 2, 0.5)
    )

app.run()