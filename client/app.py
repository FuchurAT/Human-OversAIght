import os
import sys
from utils.screen.screen import ScreenConfig, ScreenManager
from utils.log import Log
from apps.image_generator import ImageGenerator

app_dir = os.path.dirname(os.path.abspath(__file__))


def init():
    # Set log file path
    Log.set_log_path(os.path.join(app_dir, "app.log"))

    createScreens()


def createScreens():
    # Create screen configurations
    screens_config = [
        # ScreenConfig(
        #     screen_id=0,
        #     name="Video",
        #     inputs=[os.path.join(app_dir, "..", "data", "videos", "shibuya.mp4")]
        # ),
        # ScreenConfig(
        #     screen_id=0,
        #     name="Video",
        #     inputs=[os.path.join(app_dir, "..", "..", "data", "images", "plane.jpg"), os.path.join(app_dir, "..", "..", "data", "images", "pear.jpg")]
        # ),
        # ScreenConfig(
        #     screen_id=1,
        #     name="Video",
        #     grid_layout=(1, 2),
        #     screen_size=(256, 256),
        #     inputs=[["This is a text message", "Another text message"], os.path.join(app_dir, "..", "data", "images", "pear.jpg")]
        # ),
        # ScreenConfig(
        #     screen_id=1,
        #     name="Stable Diffusion 3.5",
        #     inputs=ImageGenerator,
        #     grid_layout=(1, 2),
        #     focus_border="blue",
        #     screen_size=(256, 256)
        # )
        # ScreenConfig(
        #     screen_id=0,
        #     name="Stable Diffusion 3.5",
        #     inputs=ImageGenerator,
        #     grid_layout=(1, 2),
        #     focus_border="blue",
        #     screen_size=(512, 512)
        # ),
        ScreenConfig(
            screen_id=0,
            name="Stable Diffusion 3.5",
            inputs=ImageGenerator,
            grid_layout=(1, 2),
            focus_border="blue",
            screen_size=(512, 512)
        ),
  
    ]

    # Create screen manager
    manager = ScreenManager(screens_config, show_focus=True)

    try:
        # Start all screens
        manager.start_all()

        # Keep the main thread alive
        while True:
            pass

    except KeyboardInterrupt:
        Log.info("\nShutting down: KeyboardInterrupt")
        manager.close_all()
        sys.exit(0)


if __name__ == "__main__":
    init()
