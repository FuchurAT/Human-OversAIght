#!/bin/bash

# Improved wrapper script for autostart compatibility on Ubuntu 24.04
# This script handles common autostart issues with conda environments
# and displays on screen 2 for debugging

# Set script directory as working directory
cd "$(dirname "$0")" || exit 1

# Log file for debugging autostart issues
LOG_FILE="$HOME/.human_oversaight_autostart.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_message "Starting Human OversAIght inference script"

# Function to setup multi-monitor environment
setup_display() {
    log_message "Setting up multi-monitor display for screen 2..."
    
    # Check if we have multiple monitors
    if ! xrandr --listmonitors | grep -q "Monitor 1"; then
        log_message "Single monitor detected, using default display"
        return 0
    fi
    
    # Get monitor information
    log_message "Detected monitors:"
    xrandr --listmonitors | tee -a "$LOG_FILE"
    
    # Set DISPLAY to ensure we're using the correct X server
    export DISPLAY=:0
    
    # Check if screen 2 exists and is active
    if xrandr --listmonitors | grep -q "Monitor 1"; then
        log_message "Multi-monitor setup detected"
        
        # Get the name of the second monitor
        local monitor2=$(xrandr --listmonitors | grep "Monitor 1" | awk '{print $4}')
        if [ -n "$monitor2" ]; then
            log_message "Second monitor: $monitor2"
            export MONITOR2="$monitor2"
        fi
    fi
    
    return 0
}

# Function to start visible terminal on screen 2
start_visible_terminal() {
    log_message "Starting visible terminal on screen 2..."
    
    # Check if we're in a desktop environment
    if [ -z "$DISPLAY" ]; then
        log_message "No display available, cannot start visible terminal"
        return 1
    fi
    
    # Try to start a new terminal window on screen 2
    local terminal_cmd=""
    
    # Detect available terminal emulators
    if command -v gnome-terminal > /dev/null 2>&1; then
        terminal_cmd="gnome-terminal"
        if [ -n "$MONITOR2" ]; then
            # Position window on second monitor
            terminal_cmd="gnome-terminal --geometry=120x40+1920+0 --working-directory=$(pwd)"
        fi
    elif command -v konsole > /dev/null 2>&1; then
        terminal_cmd="konsole"
        if [ -n "$MONITOR2" ]; then
            terminal_cmd="konsole --geometry 120x40+1920+0 --workdir $(pwd)"
        fi
    elif command -v xterm > /dev/null 2>&1; then
        terminal_cmd="xterm"
        if [ -n "$MONITOR2" ]; then
            terminal_cmd="xterm -geometry 120x40+1920+0 -e 'cd $(pwd) && bash'"
        fi
    else
        log_message "No suitable terminal emulator found"
        return 1
    fi
    
    if [ -n "$terminal_cmd" ]; then
        log_message "Starting terminal: $terminal_cmd"
        # Start terminal in background
        $terminal_cmd &
        sleep 2
        
        # Check if terminal started successfully
        if pgrep -f "$(basename $terminal_cmd)" > /dev/null; then
            log_message "Terminal started successfully"
            return 0
        else
            log_message "Failed to start terminal"
            return 1
        fi
    fi
    
    return 1
}

# Wait for system to be ready (more intelligent than fixed sleep)
wait_for_system() {
    log_message "Waiting for system to be ready..."
    
    # Wait for user to be logged in
    log_message "Checking for user session..."
    while ! ps aux | grep -q "gnome.*session\|gnome-shell\|kdeinit\|xfce4-session"; do
        log_message "Waiting for user session... (any GNOME session process)"
        sleep 2
    done
    log_message "User session detected"
    
    # Wait for X server to be ready
    log_message "Checking X server..."
    
    # Ensure DISPLAY is set to the main display
    export DISPLAY=:0
    log_message "Set DISPLAY to :0"
    
    # Also try to set XAUTHORITY if not set
    if [ -z "$XAUTHORITY" ] && [ -f "/run/user/$(id -u)/gdm/Xauthority" ]; then
        export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
        log_message "Set XAUTHORITY to $XAUTHORITY"
    fi
    
    local max_wait=60
    local wait_count=0
    while ! xset q > /dev/null 2>&1 && [ $wait_count -lt $max_wait ]; do
        log_message "Waiting for X server... (attempt $((wait_count/2 + 1))/$((max_wait/2)))"
        log_message "DISPLAY=$DISPLAY, XAUTHORITY=$XAUTHORITY"
        sleep 2
        wait_count=$((wait_count + 2))
    done
    
    if [ $wait_count -ge $max_wait ]; then
        log_message "ERROR: X server not ready after $max_wait seconds"
        log_message "Trying alternative X server check..."
        # Try alternative method
        if xdpyinfo > /dev/null 2>&1; then
            log_message "X server accessible via xdpyinfo"
        else
            log_message "ERROR: X server not accessible via any method"
            return 1
        fi
    fi
    log_message "X server is ready"
    
    # Wait for conda to be available
    log_message "Checking conda availability..."
    
    # Try to find and source conda if not in PATH
    if ! command -v conda > /dev/null 2>&1; then
        log_message "Conda not in PATH, searching for installation..."
        
        # Common conda locations
        local conda_paths=(
            "$HOME/anaconda3"
            "$HOME/miniconda3"
            "$HOME/.conda"
            "/opt/conda"
            "/usr/local/conda"
        )
        
        local conda_found=false
        for path in "${conda_paths[@]}"; do
            if [ -f "$path/etc/profile.d/conda.sh" ]; then
                log_message "Found conda at: $path"
                source "$path/etc/profile.d/conda.sh"
                export CONDA_ROOT="$path"
                conda_found=true
                break
            fi
        done
        
        if [ "$conda_found" = false ]; then
            log_message "ERROR: Conda not found in common locations"
            return 1
        fi
    fi
    
    # Now check if conda is available
    wait_count=0
    while ! command -v conda > /dev/null 2>&1 && [ $wait_count -lt $max_wait ]; do
        log_message "Waiting for conda to be available... (attempt $((wait_count/2 + 1))/$((max_wait/2)))"
        sleep 2
        wait_count=$((wait_count + 2))
    done
    
    if [ $wait_count -ge $max_wait ]; then
        log_message "ERROR: Conda not available after $max_wait seconds"
        return 1
    fi
    log_message "Conda is available"
    
    log_message "System ready, X server and conda available"
    return 0
}

# Function to find conda installation
find_conda() {
    log_message "Searching for conda installation..."
    
    # Common conda locations
    local conda_paths=(
        "$HOME/anaconda3"
        "$HOME/miniconda3"
        "$HOME/.conda"
        "/opt/conda"
        "/usr/local/conda"
    )
    
    # Also check PATH
    if command -v conda > /dev/null 2>&1; then
        local conda_path=$(which conda)
        log_message "Conda found in PATH: $conda_path"
        return 0
    fi
    
    # Check common locations
    for path in "${conda_paths[@]}"; do
        if [ -f "$path/etc/profile.d/conda.sh" ]; then
            log_message "Conda found at: $path"
            export CONDA_ROOT="$path"
            return 0
        fi
    done
    
    log_message "ERROR: Conda not found in common locations"
    return 1
}

# Function to setup conda environment
setup_conda() {
    log_message "Setting up conda environment..."
    
    # Source conda if not in PATH
    if ! command -v conda > /dev/null 2>&1; then
        if [ -n "$CONDA_ROOT" ]; then
            source "$CONDA_ROOT/etc/profile.d/conda.sh"
            log_message "Sourced conda from: $CONDA_ROOT"
        else
            log_message "ERROR: Cannot setup conda - not found"
            return 1
        fi
    fi
    
    # Initialize conda for this shell session
    eval "$(conda shell.bash hook)"
    log_message "Initialized conda shell hook"
    
    # Check if cnn-detection environment exists
    if ! conda env list | grep -q "cnn-detection"; then
        log_message "ERROR: cnn-detection environment not found"
        log_message "Available environments:"
        conda env list | tee -a "$LOG_FILE"
        return 1
    fi
    
    # Activate environment
    log_message "Activating cnn-detection environment..."
    conda activate cnn-detection
    
    if [ "$(conda info --envs | grep '*' | awk '{print $1}')" != "cnn-detection" ]; then
        log_message "ERROR: Failed to activate cnn-detection environment"
        return 1
    fi
    
    log_message "Successfully activated cnn-detection environment"
    return 0
}

# Function to check dependencies
check_dependencies() {
    log_message "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python > /dev/null 2>&1; then
        log_message "ERROR: Python not found"
        return 1
    fi
    
    # Check if required files exist
    if [ ! -f "run_multi_apps.py" ]; then
        log_message "ERROR: run_multi_apps.py not found in current directory"
        return 1
    fi
    
    log_message "Dependencies check passed"
    return 0
}

# Main execution
main() {
    log_message "=== Human OversAIght Autostart Started ==="
    
    # Wait for system to be ready
    if ! wait_for_system; then
        log_message "ERROR: System not ready, exiting"
        exit 1
    fi
    
    # Setup display and multi-monitor
    if ! setup_display; then
        log_message "WARNING: Display setup failed, continuing anyway"
    fi
    
    # Start visible terminal on screen 2
    if ! start_visible_terminal; then
        log_message "WARNING: Failed to start visible terminal, continuing in background"
    fi
    
    # Find conda installation
    if ! find_conda; then
        log_message "ERROR: Conda not found, exiting"
        exit 1
    fi
    
    # Setup conda environment
    if ! setup_conda; then
        log_message "ERROR: Failed to setup conda, exiting"
        exit 1
    fi
    
    # Check dependencies
    if ! check_dependencies; then
        log_message "ERROR: Dependencies check failed, exiting"
        exit 1
    fi
    
    # Run the inference script
    log_message "Starting inference with arguments: $*"
    log_message "Current directory: $(pwd)"
    log_message "Python path: $(which python)"
    log_message "Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
    
    # Run the script with all arguments passed through
    if python run_multi_apps.py "$@"; then
        log_message "Inference script completed successfully"
    else
        log_message "ERROR: Inference script failed with exit code $?"
        exit 1
    fi
}

# Run main function with all arguments
main "$@" 