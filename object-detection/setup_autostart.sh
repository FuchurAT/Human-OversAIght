#!/bin/bash

# Setup script for Human OversAIght autostart on Ubuntu 24.04
# This script configures autostart with proper multi-monitor support

set -e

echo "=== Human OversAIght Autostart Setup ==="
echo "This script will configure autostart for Ubuntu 24.04"
echo ""

# Check if running as user (not root)
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Please run this script as a regular user, not as root"
    exit 1
fi

# Get current user
CURRENT_USER=$(whoami)
echo "Setting up autostart for user: $CURRENT_USER"

# Check if we're in the right directory
if [ ! -f "run_inference.sh" ]; then
    echo "ERROR: Please run this script from the object-detection directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x run_inference.sh
chmod +x setup_autostart.sh

# Check for autostart directory
AUTOSTART_DIR="$HOME/.config/autostart"
if [ ! -d "$AUTOSTART_DIR" ]; then
    echo "Creating autostart directory: $AUTOSTART_DIR"
    mkdir -p "$AUTOSTART_DIR"
fi

# Copy desktop entry to autostart
echo "Installing autostart entry..."
cp "Human-OversAIght-Autostart.desktop" "$AUTOSTART_DIR/"

# Make desktop entry executable
chmod +x "$AUTOSTART_DIR/Human-OversAIght-Autostart.desktop"

# Check for multi-monitor setup
echo ""
echo "Checking multi-monitor setup..."
if command -v xrandr > /dev/null 2>&1; then
    echo "Detected monitors:"
    xrandr --listmonitors
    
    # Check if we have multiple monitors
    MONITOR_COUNT=$(xrandr --listmonitors | grep -c "Monitor")
    if [ "$MONITOR_COUNT" -gt 1 ]; then
        echo "Multi-monitor setup detected ($MONITOR_COUNT monitors)"
        echo "Script will attempt to display on screen 2"
    else
        echo "Single monitor setup detected"
        echo "Script will use default display"
    fi
else
    echo "WARNING: xrandr not available, cannot detect monitor setup"
fi

# Check conda installation
echo ""
echo "Checking conda installation..."
if command -v conda > /dev/null 2>&1; then
    echo "Conda found: $(which conda)"
    echo "Available environments:"
    conda env list | grep -E "^\*|^cnn-detection"
    
    if conda env list | grep -q "cnn-detection"; then
        echo "✓ cnn-detection environment found"
    else
        echo "⚠ WARNING: cnn-detection environment not found"
        echo "  Please create it first: conda create -n cnn-detection python=3.8"
    fi
else
    echo "⚠ WARNING: Conda not found in PATH"
    echo "  The script will attempt to find it automatically"
fi

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."
if [ -f "run_multi_apps.py" ]; then
    echo "✓ run_multi_apps.py found"
else
    echo "⚠ WARNING: run_multi_apps.py not found"
fi

# Test the script
echo ""
echo "Testing the inference script..."
if ./run_inference.sh --help > /dev/null 2>&1; then
    echo "✓ Script test successful"
else
    echo "⚠ WARNING: Script test failed"
    echo "  This may indicate missing dependencies"
fi

# Create log directory
LOG_DIR="$HOME/.human_oversaight_logs"
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

# Final instructions
echo ""
echo "=== Setup Complete ==="
echo "Autostart entry installed to: $AUTOSTART_DIR/Human-OversAIght-Autostart.desktop"
echo "Log files will be written to: $HOME/.human_oversaight_autostart.log"
echo ""
echo "To enable autostart:"
echo "1. Log out and log back in, or restart your system"
echo "2. The script will start automatically after login"
echo "3. Check the log file for any errors: tail -f $HOME/.human_oversaight_autostart.log"
echo ""
echo "To disable autostart:"
echo "  rm '$AUTOSTART_DIR/Human-OversAIght-Autostart.desktop'"
echo ""
echo "To test manually:"
echo "  ./run_inference.sh"
echo ""
echo "To view logs:"
echo "  tail -f $HOME/.human_oversaight_autostart.log" 