import os
import time
import requests
import json
import asyncio
from typing import List
from utils.screen.screen import ScreenApplication, Screen
from utils.log import Log
import threading

app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(app_dir, '..', '..'))
decoder = json.JSONDecoder()


class ImageGenerator(ScreenApplication):
    """Image generator application for screen display"""

    def __init__(self, screen: Screen):
        self.current_prompt = ""
        self.current_image_path = ""
        self.running = True
        super().__init__(screen)
        # Start the async loop in a separate thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_async_loop)
        self.thread.daemon = True
        self.thread.start()

    def _run_async_loop(self):
        """Run the async loop in a separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_loop())

    async def _run_loop(self):
        while self.running:
            try:
                # Generate prompt
                prompt_response = requests.get("http://localhost:8811/generate", params={
                                               "prompt": "Return prompt for image generation that shows post apocalyptic city after climate change made the earth inhabitle."})

                json_response = decoder.decode(
                    decoder.decode(prompt_response.text))

                if prompt_response.status_code == 200:
                    self.current_prompt = json_response["generated_text"]
                    self.current_image_path = ""
                    self.update()

                    # Generate image
                    image_response = requests.get("http://localhost:8812/generate", params={
                        "prompt": self.current_prompt,
                        "width": 256,
                        "height": 256,
                        "randomize_seed": True
                    })

                    if image_response.status_code == 200:
                        json_response = decoder.decode(
                            decoder.decode(image_response.text))
                        image_path = json_response["image_path"]
                        
                        if image_path:
                            # Convert /data path to project root relative path
                            if image_path.startswith('/data'):
                                image_path = os.path.join(project_root, image_path[1:])
                            self.current_image_path = image_path
                            self.update()
            except Exception as e:
                Log.error(f"ImageGenerator error: {e}")

            await asyncio.sleep(60)

    def input(self) -> List[str]:
        return [[self.current_prompt], self.current_image_path] if self.current_prompt else []

    def close(self):
        """Clean up resources when closing"""
        self.running = False
        if self.loop and self.loop.is_running():
            self.loop.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
