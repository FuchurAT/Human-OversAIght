import os
import time
import requests
import json
import asyncio
from typing import List
from utils.screen.screen import ScreenApplication, Screen
from utils.log import Log
import threading
import glob

app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(app_dir, '..', '..'))
decoder = json.JSONDecoder()

class GradcamGenerator(ScreenApplication):
    """Gradcam video generator application for screen display"""

    def __init__(self, screen: Screen):
        self.current_video_path = ""
        self.current_processed_video = ""
        self.running = True
        self.video_queue = []
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

    def _load_video_queue(self):
        """Load all videos from the data/videos directory"""
        videos_dir = os.path.join(project_root, 'data', 'videos')
        if not os.path.exists(videos_dir):
            Log.error(f"Videos directory not found: {videos_dir}")
            return

        video_files = glob.glob(os.path.join(videos_dir, '*'))
        self.video_queue = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        Log.info(f"Loaded {len(self.video_queue)} videos into queue")

    async def _process_next_video(self):
        """Process the next video in the queue"""
        if not self.video_queue:
            self._load_video_queue()
            if not self.video_queue:
                return

        video_path = self.video_queue.pop(0)
        self.current_video_path = video_path
        self.update()

        try:
            # Convert video path to /data/videos/FILE format
            video_filename = os.path.basename(video_path)
            api_video_path = f"/data/videos/{video_filename}"

            # Request gradcam processing
            response = requests.get("http://localhost:8813/generate", params={
                "input_path": api_video_path
            })

            if response.status_code == 200:
                json_response = decoder.decode(decoder.decode(response.text))
                processed_video = json_response.get("video_path")
                
                if processed_video:
                    # Convert /data path to project root relative path if needed
                    if processed_video.startswith('/data'):
                        processed_video = os.path.join(project_root, processed_video[1:])
                    self.current_processed_video = processed_video
                    self.update()
            else:
                Log.error(f"Failed to process video: {response.status_code}")

        except Exception as e:
            Log.error(f"Error processing video: {e}")

    async def _run_loop(self):
        while self.running:
            try:
                if not self.current_processed_video:
                    await self._process_next_video()
                await asyncio.sleep(0.1)
            except Exception as e:
                Log.error(f"GradcamGenerator error: {e}")
                await asyncio.sleep(1)

    def input(self) -> List[str]:
        if self.current_processed_video:
            return [self.current_processed_video]
        return []

    def close(self):
        """Clean up resources when closing"""
        self.running = False
        if self.loop and self.loop.is_running():
            self.loop.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1) 