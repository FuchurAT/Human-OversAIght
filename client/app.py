import os
import sys
from utils.screen.screen import ScreenConfig, ScreenManager
from utils.log import Log
from apps.object_detection_app import ObjectDetectionApp

app_dir = os.path.dirname(os.path.abspath(__file__))


def init():
    # Set log file path
    Log.set_log_path(os.path.join(app_dir, "app.log"))

    createScreens()


def createScreens():
    # Create screen configurations
    screens_config = [
        # Placeholder screen for testing
        ScreenConfig(
            screen_id=0,
            name="Welcome",
            inputs=["E:/Projects/human-oversaight/client/apps/object-detection/videos/us_capitol.mp4"],
            grid_layout=(1, 1),
            focus_border="blue",
            screen_size=(800, 600)
        ),
        # Object Detection screen
        ScreenConfig(
            screen_id=1,
            name="Object Detection",
            inputs=ObjectDetectionApp,
            grid_layout=(1, 1),
            focus_border="red",
            screen_size=(800, 600)
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
