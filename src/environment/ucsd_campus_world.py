import pygame
from pathlib import Path

class UCSDCampusWorld:
    """
    A 2D environment that uses the extracted UCSD campus map (from page 1 of the PDF)
    as the background. Interactive landmarks and paths are overlaid for agent navigation.
    """
    def __init__(self, width=5760, height=3240):
        self.width = width
        self.height = height

        # Initialize pygame and create an enlarged window
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("UCSD Campus World - Exact Replica from PDF")

        # Determine the assets folder (using parents[2] for robustness)
        self.assets_path = Path(__file__).parents[1] / "assets"
        map_file = self.assets_path / "ucsd_map.jpg"
        print(f"Looking for map at: {map_file}")
        if map_file.exists():
            self.terrain_texture = pygame.image.load(str(map_file)).convert_alpha()
            self.terrain_texture = pygame.transform.scale(self.terrain_texture, (self.width, self.height))
        else:
            print("Map file not found! Using fallback background.")
            # Fallback: fill with a distinct color so it's obvious there's an issue
            self.terrain_texture = pygame.Surface((self.width, self.height))
            self.terrain_texture.fill((200, 200, 200))  # light gray

        # Define interactive landmarks (adjust positions as needed based on the actual map)
        self.landmarks = {
            "Geisel Library": {"position": (960, 540), "radius": 40},
            "Price Center":    {"position": (1300, 700), "radius": 40},
            "RIMAC":           {"position": (600, 400), "radius": 40},
            "CSE Building":    {"position": (1600, 500), "radius": 35},
            "Cognitive Science": {"position": (1300, 480), "radius": 35}
        }

        # Define simple paths connecting these landmarks for visual overlay
        lm_positions = [lm["position"] for lm in self.landmarks.values()]
        self.paths = []
        if lm_positions:
            n = len(lm_positions)
            for i in range(n):
                start = lm_positions[i]
                end = lm_positions[(i + 1) % n]
                self.paths.append((start, end))

    def render(self):
        # Draw the campus map (background image)
        self.screen.blit(self.terrain_texture, (0, 0))
        
        # Overlay landmarks (draw circles and labels on the map)
        font = pygame.font.SysFont("Arial", 24, bold=True)
        for name, data in self.landmarks.items():
            pos = data["position"]
            pygame.draw.circle(self.screen, (0, 0, 255), pos, data["radius"], 3)
            text = font.render(name, True, (0, 0, 0))
            text_rect = text.get_rect(center=(pos[0], pos[1] - data["radius"] - 30))
            self.screen.blit(text, text_rect)
        
        # Draw paths between landmarks (simple red line for visual effect)
        for (start, end) in self.paths:
            pygame.draw.line(self.screen, (255, 0, 0), start, end, 2)
        
        pygame.display.flip()

    def run(self):
        """Main loop to display and update the campus world."""
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.render()
            clock.tick(30)
        pygame.quit()

if __name__ == "__main__":
    world = UCSDCampusWorld()
    world.run()