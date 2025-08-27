#!/usr/bin/env python3
"""
Test script for next video button action
"""

import time
import logging
from application.button_handler import ButtonHandler
from application.multi_app_manager import MultiAppManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_next_video_action():
    """Test that the next video button action doesn't freeze"""
    print("Testing next video button action...")
    
    try:
        # Create a mock button handler
        button_handler = ButtonHandler()
        
        # Test the next video action without any app instances
        print("Testing next video action with no app instances...")
        button_handler._action_next_video()
        print("✓ Next video action completed without freezing")
        
        # Test with mock app instances
        print("Testing next video action with mock app instances...")
        mock_app = type('MockApp', (), {
            'app_id': 'test_app',
            'signal_next_video': lambda: print("Mock app signal_next_video called")
        })()
        
        button_handler.add_app_instance('test_app', mock_app)
        button_handler._action_next_video()
        print("✓ Next video action with mock app completed without freezing")
        
        print("All tests passed! Next video button action is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_next_video_action() 