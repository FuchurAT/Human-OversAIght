import logging
from pathlib import Path
import argparse
from video_inference_app import VideoInferenceApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def main():
    parser = argparse.ArgumentParser(description="Run real-time inference on a video file with YOLO.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output", type=str, default="", help="Path to save the output video") # "output/realtime_detection.mp4"
    parser.add_argument("--box-threshold", type=float, default=0.1, help="Minimum confidence for showing a detection box")
    parser.add_argument("--show-legend", action="store_true", help="Show legend with detected classes and confidence scores")
    args = parser.parse_args()
 
    # Get the directory of the current script and construct relative path to model
    current_dir = Path(__file__).parent
    model_path = current_dir / "runs" / "train" / "weights" / "best.pt"
    
    try:
        app = VideoInferenceApp(args.video_path, model_path, args.output, args.box_threshold, args.show_legend)
        app.run()
    except Exception as e:
        logging.error(f"Error running inference: {e}")
        return
 
if __name__ == "__main__":
    main() 