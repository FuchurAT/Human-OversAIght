#!/usr/bin/env python3
"""
Simple script to run the multi-application object detection system.
This is an alternative to using the main inference.py with --mode multi.
"""

import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from application.multi_app_manager import MultiAppManager
from config.config import LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format']
)

def main():
    """Run the multi-application system"""
    try:
        logging.info("Starting multi-application object detection system...")
        
        # Create and run the manager
        manager = MultiAppManager()
        manager.run()
        
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Error running multi-application system: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
