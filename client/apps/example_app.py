import os
from typing import List
from utils.screen.screen import ScreenApplication, Screen

app_dir = os.path.dirname(os.path.abspath(__file__))


class ExampleApp(ScreenApplication):
    def __init__(self, screen: Screen):
        super().__init__(screen)

    def input() -> List[str]:
        return [
            os.path.join(app_dir, "..", "..", "data", "images", "plane.jpg"),
            os.path.join(app_dir, "..", "..", "data", "images", "pear.jpg"),
        ]
