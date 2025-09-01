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
    log_message "Setting up multi-monitor display environment..."
    
    # Set DISPLAY to ensure we're using the correct X server
    export DISPLAY=:0
    
    # Wait for X server to be fully ready
    local max_wait=30
    local wait_count=0
    while ! xrandr --listmonitors > /dev/null 2>&1 && [ $wait_count -lt $max_wait ]; do
        log_message "Waiting for X server to be ready for monitor detection... (attempt $((wait_count + 1))/$max_wait)"
        sleep 1
        wait_count=$((wait_count + 1))
    done
    
    if [ $wait_count -ge $max_wait ]; then
        log_message "ERROR: X server not ready for monitor detection after $max_wait seconds"
        return 1
    fi
    
    # Get detailed monitor information
    log_message "Detecting monitors..."
    local monitor_info=$(xrandr --listmonitors 2>/dev/null)
    if [ $? -ne 0 ]; then
        log_message "ERROR: Failed to get monitor information from xrandr"
        return 1
    fi
    
    log_message "Raw monitor information:"
    echo "$monitor_info" | tee -a "$LOG_FILE"
    
    # Parse monitor information more robustly
    local monitor_count=0
    local monitors=()
    local monitor_names=()
    local monitor_positions=()
    
    # Parse each line of monitor information
    while IFS= read -r line; do
        if [[ "$line" =~ ^[[:space:]]*([0-9]+):[[:space:]]+([^[:space:]]+)[[:space:]]+([0-9]+)x([0-9]+)[[:space:]]+([+-][0-9]+)/([+-][0-9]+) ]]; then
            local monitor_num="${BASH_REMATCH[1]}"
            local monitor_name="${BASH_REMATCH[2]}"
            local monitor_width="${BASH_REMATCH[3]}"
            local monitor_height="${BASH_REMATCH[4]}"
            local monitor_x="${BASH_REMATCH[5]}"
            local monitor_y="${BASH_REMATCH[6]}"
            
            monitors+=("$monitor_num")
            monitor_names+=("$monitor_name")
            monitor_positions+=("${monitor_x}${monitor_y}")
            
            log_message "Monitor $monitor_num: $monitor_name (${monitor_width}x${monitor_height}) at (${monitor_x},${monitor_y})"
            monitor_count=$((monitor_count + 1))
        fi
    done <<< "$monitor_info"
    
    log_message "Detected $monitor_count monitor(s)"
    
    # Export monitor information for other functions
    export MONITOR_COUNT="$monitor_count"
    export MONITOR_NAMES="${monitor_names[*]}"
    export MONITOR_POSITIONS="${monitor_positions[*]}"
    
    # Set specific monitor variables for backward compatibility
    if [ "$monitor_count" -ge 2 ]; then
        export MONITOR2="${monitor_names[1]}"
        log_message "Second monitor: $MONITOR2"
    fi
    
    if [ "$monitor_count" -ge 3 ]; then
        export MONITOR3="${monitor_names[2]}"
        log_message "Third monitor: $MONITOR3"
    fi
    
    # Validate that monitors are actually connected and active
    log_message "Validating monitor connections..."
    local active_monitors=0
    
    for monitor_name in "${monitor_names[@]}"; do
        if xrandr --query | grep -q "^$monitor_name connected"; then
            log_message "✓ Monitor $monitor_name is connected and active"
            active_monitors=$((active_monitors + 1))
        else
            log_message "⚠ Monitor $monitor_name is not connected or inactive"
        fi
    done
    
    if [ "$active_monitors" -eq 0 ]; then
        log_message "ERROR: No active monitors detected"
        return 1
    fi
    
    log_message "Monitor setup completed: $active_monitors active monitor(s) out of $monitor_count detected"
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
    
    # Check if we have a second monitor
    if [ -z "$MONITOR2" ]; then
        log_message "No second monitor detected, starting terminal on primary display"
    else
        log_message "Second monitor detected: $MONITOR2"
    fi
    
    # Try to start a new terminal window on screen 2
    local terminal_cmd=""
    local geometry=""
    
    # Calculate position for second monitor if available
    if [ -n "$MONITOR2" ]; then
        # Get the actual position of screen 2 from xrandr
        local screen2_info=$(xrandr --query | grep "^$MONITOR2" | grep -o "[+-][0-9]*/[+-][0-9]*" | head -1)
        if [ -n "$screen2_info" ]; then
            local screen2_x=$(echo "$screen2_info" | cut -d'/' -f1 | sed 's/+//')
            local screen2_y=$(echo "$screen2_info" | cut -d'/' -f2 | sed 's/+//')
            if [ -n "$screen2_x" ] && [ -n "$screen2_y" ]; then
                geometry="120x40+${screen2_x}+${screen2_y}"
                log_message "Calculated geometry for screen 2: $geometry"
            fi
        fi
    fi
    
    # If we couldn't calculate position, use default second monitor position
    if [ -z "$geometry" ] && [ -n "$MONITOR2" ]; then
        geometry="120x40+1920+0"
        log_message "Using default geometry for screen 2: $geometry"
    fi
    
    # Detect available terminal emulators
    if command -v gnome-terminal > /dev/null 2>&1; then
        terminal_cmd="gnome-terminal"
        if [ -n "$geometry" ]; then
            terminal_cmd="gnome-terminal --geometry=$geometry --working-directory=$(pwd)"
        fi
    elif command -v konsole > /dev/null 2>&1; then
        terminal_cmd="konsole"
        if [ -n "$geometry" ]; then
            terminal_cmd="konsole --geometry $geometry --workdir $(pwd)"
        fi
    elif command -v xterm > /dev/null 2>&1; then
        terminal_cmd="xterm"
        if [ -n "$geometry" ]; then
            terminal_cmd="xterm -geometry $geometry -e 'cd $(pwd) && bash'"
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

# Function to clean up OBS lock files and state
cleanup_obs_state() {
    log_message "Cleaning up OBS state files to prevent unclean shutdown messages..."
    
    # Common OBS state directories
    local obs_dirs=(
        "$HOME/.config/obs-studio"
        "$HOME/.local/share/obs-studio"
        "$HOME/.cache/obs-studio"
    )
    
    # Files that might cause unclean shutdown messages
    local cleanup_files=(
        "global.ini"
        "basic/profiles/Untitled/basic.ini"
        "basic/profiles/Untitled/profanity_filter.txt"
        "basic/profiles/Untitled/service.json"
        "basic/profiles/Untitled/streamEncoder.json"
        "basic/profiles/Untitled/recordEncoder.json"
        "basic/profiles/Untitled/outputs.json"
        "basic/profiles/Untitled/audio.json"
        "basic/profiles/Untitled/video.json"
        "basic/profiles/Untitled/hotkeys.json"
        "basic/profiles/Untitled/advanced.json"
        "basic/profiles/Untitled/recordEncoder.json"
        "basic/profiles/Untitled/streamEncoder.json"
        "basic/profiles/Untitled/outputs.json"
        "basic/profiles/Untitled/audio.json"
        "basic/profiles/Untitled/video.json"
        "basic/profiles/Untitled/hotkeys.json"
        "basic/profiles/Untitled/advanced.json"
    )
    
    for dir in "${obs_dirs[@]}"; do
        if [ -d "$dir" ]; then
            log_message "Checking directory: $dir"
            for file in "${cleanup_files[@]}"; do
                local full_path="$dir/$file"
                if [ -f "$full_path" ]; then
                    # Backup the file before removing
                    if [ -f "${full_path}.backup" ]; then
                        rm -f "${full_path}.backup"
                    fi
                    mv "$full_path" "${full_path}.backup"
                    log_message "Backed up and removed: $file"
                fi
            done
        fi
    done
    
    # Also check for any running OBS processes and kill them cleanly
    if pgrep -f "obs" > /dev/null; then
        log_message "Found running OBS processes, attempting clean shutdown..."
        pkill -TERM -f "obs" 2>/dev/null
        sleep 2
        
        # Force kill if still running
        if pgrep -f "obs" > /dev/null; then
            log_message "Force killing remaining OBS processes..."
            pkill -KILL -f "obs" 2>/dev/null
            sleep 1
        fi
    fi
    
    log_message "OBS state cleanup completed"
}

# Function to start OBS Studio with specific scene collection
start_obs() {
    log_message "Starting OBS Studio with scene collection: opsroom-1..."
    
    # Clean up any existing OBS state first
    cleanup_obs_state
    
    # Log current monitor setup for OBS
    log_message "Current monitor setup for OBS:"
    log_message "  Total monitors: $MONITOR_COUNT"
    log_message "  Monitor names: $MONITOR_NAMES"
    log_message "  Monitor positions: $MONITOR_POSITIONS"
    
    # Check if OBS Studio is installed
    if ! command -v obs-studio > /dev/null 2>&1; then
        log_message "WARNING: OBS Studio not found in PATH, searching for installation..."
        
        # Common OBS installation locations
        local obs_paths=(
            "/usr/bin/obs"
            "/usr/bin/obs-studio"
            "/usr/local/bin/obs"
            "/usr/local/bin/obs-studio"
            "$HOME/.local/bin/obs"
            "$HOME/.local/bin/obs-studio"
            "/snap/bin/obs-studio"
            "/opt/obs-studio/bin/obs"
            "/opt/obs-studio/bin/obs-studio"
        )
        
        local obs_found=false
        for path in "${obs_paths[@]}"; do
            if [ -x "$path" ]; then
                log_message "Found OBS at: $path"
                export OBS_PATH="$path"
                obs_found=true
                break
            fi
        done
        
        # If still not found, check app manager installations
        if [ "$obs_found" = false ]; then
            log_message "Checking app manager installations..."
            
            # Check for .desktop files to find the actual executable
            local desktop_file=$(find /usr/share/applications /usr/local/share/applications $HOME/.local/share/applications -name "*obs*" -type f 2>/dev/null | head -1)
            if [ -n "$desktop_file" ]; then
                log_message "Found OBS desktop file: $desktop_file"
                # Extract Exec= line from desktop file
                local exec_line=$(grep "^Exec=" "$desktop_file" | head -1 | cut -d'=' -f2 | cut -d' ' -f1)
                if [ -n "$exec_line" ] && [ -x "$exec_line" ]; then
                    log_message "Found OBS executable from desktop file: $exec_line"
                    export OBS_PATH="$exec_line"
                    obs_found=true
                fi
            fi
            
            # Check for snap installations
            if [ "$obs_found" = false ] && command -v snap > /dev/null 2>&1; then
                if snap list | grep -q "obs"; then
                    log_message "Found OBS via snap, using snap run"
                    export OBS_PATH="snap run obs-studio"
                    obs_found=true
                fi
            fi
            
            # Check for flatpak installations
            if [ "$obs_found" = false ] && command -v flatpak > /dev/null 2>&1; then
                if flatpak list | grep -q "obs"; then
                    log_message "Found OBS via flatpak, using flatpak run"
                    export OBS_PATH="flatpak run com.obsproject.Studio"
                    obs_found=true
                fi
            fi
        fi
        
        if [ "$obs_found" = false ]; then
            log_message "WARNING: OBS Studio not found, skipping OBS startup"
            return 1
        fi
    else
        export OBS_PATH="obs-studio"
    fi
    
    # Check if OBS is already running
    if pgrep -f "obs" > /dev/null; then
        log_message "OBS Studio is already running, skipping startup"
        return 0
    fi
    
    # Start OBS with the opsroom-1 scene collection
    log_message "Starting OBS Studio with scene collection: opsroom-1"

    # Set graphics options for NVIDIA GPUs
    export __GL_SYNC_TO_VBLANK=0
    export __VDPAU_NVIDIA_SYNC_TO_VBLANK=0
    
    # Try NVIDIA backend first, fallback to X11 if needed
    local obs_started=false
    
    # First attempt: Try with NVIDIA backend and force start flags
    log_message "Attempting to start OBS with NVIDIA backend and force start flags..."
    if [ -n "$OBS_PATH" ]; then
        $OBS_PATH --collection "opsroom-1" --gl-backend nvidia --safe-mode --disable-shutdown-check --minimize-to-tray --startpreview &
        local obs_pid=$!
        sleep 3
        
        # Check if OBS started successfully
        if kill -0 $obs_pid 2>/dev/null; then
            log_message "OBS Studio started successfully with NVIDIA backend (PID: $obs_pid)"
            obs_started=true
        else
            log_message "NVIDIA backend failed, trying X11 backend..."
        fi
    fi
    
    # Second attempt: Fallback to X11 backend if NVIDIA failed
    if [ "$obs_started" = false ] && [ -n "$OBS_PATH" ]; then
        log_message "Starting OBS with X11 backend fallback and force start flags..."
        $OBS_PATH --collection "opsroom-1" --gl-backend x11 --safe-mode --disable-shutdown-check --minimize-to-tray --startpreview &
        local obs_pid=$!
        sleep 3
        
        # Check if OBS started successfully
        if kill -0 $obs_pid 2>/dev/null; then
            log_message "OBS Studio started successfully with X11 backend (PID: $obs_pid)"
            obs_started=true
        else
            log_message "WARNING: Failed to start OBS Studio with any backend"
            return 1
        fi
    fi
    
    # Final check and return
    if [ "$obs_started" = true ]; then
        log_message "OBS Studio started successfully"
        return 0
    else
        log_message "ERROR: Failed to start OBS Studio with any backend"
        return 1
    fi
}

# Function to position OBS preview projector on screen 3
position_preview_projector() {
    log_message "Positioning OBS preview projector on screen 3..."
    
    # Wait a bit for OBS to fully initialize
    sleep 5
    
    # Check if we have screen 3
    local screen_count=$(xrandr --listmonitors | grep -c "Monitor")
    log_message "Detected $screen_count monitor(s)"
    
    if [ "$screen_count" -lt 3 ]; then
        log_message "WARNING: Only $screen_count monitor(s) detected, cannot position on screen 3"
        return 1
    fi
    
    # Get screen 3 information (assuming 0-indexed, so screen 3 is Monitor 2)
    local screen3_info=$(xrandr --listmonitors | grep "Monitor 2" | awk '{print $4}')
    if [ -z "$screen3_info" ]; then
        log_message "WARNING: Screen 3 information not found"
        return 1
    fi
    
    log_message "Screen 3: $screen3_info"
    
    # Try to position the preview projector window on screen 3
    # First, find the OBS preview projector window
    local obs_preview_window=$(xdotool search --name "Preview" 2>/dev/null | head -1)
    
    if [ -n "$obs_preview_window" ]; then
        log_message "Found OBS preview window: $obs_preview_window"
        
        # Get screen 3 position (assuming 1920x1080 monitors, adjust as needed)
        local screen3_x=3840  # 2 * 1920 for screen 3
        local screen3_y=0
        
        # Move the preview window to screen 3
        if command -v xdotool > /dev/null 2>&1; then
            xdotool windowmove "$obs_preview_window" "$screen3_x" "$screen3_y" 2>/dev/null
            if [ $? -eq 0 ]; then
                log_message "Successfully positioned preview projector on screen 3"
                return 0
            else
                log_message "WARNING: Failed to position preview projector with xdotool"
            fi
        else
            log_message "WARNING: xdotool not available, cannot position preview projector"
        fi
    else
        log_message "WARNING: OBS preview window not found"
    fi
    
    # Alternative method: try using wmctrl if xdotool fails
    if command -v wmctrl > /dev/null 2>&1; then
        log_message "Trying alternative positioning with wmctrl..."
        local screen3_x=3840
        local screen3_y=0
        
        # Find and move the preview window
        wmctrl -r "Preview" -e 0,"$screen3_x","$screen3_y",-1,-1 2>/dev/null
        if [ $? -eq 0 ]; then
            log_message "Successfully positioned preview projector on screen 3 using wmctrl"
            return 0
        else
            log_message "WARNING: Failed to position preview projector with wmctrl"
        fi
    fi
    
    log_message "WARNING: Could not position preview projector on screen 3"
    return 1
}

# Function to continuously monitor and maintain the preview projector
monitor_preview_projector() {
    log_message "Starting preview projector monitoring (ensures it never stops)..."
    
    # Run monitoring in background
    (
        while true; do
            # Check if OBS is still running
            if ! pgrep -f "obs" > /dev/null; then
                log_message "OBS process not found, stopping preview monitoring"
                break
            fi
            
            # Check if preview projector window exists
            local preview_window=$(xdotool search --name "Preview" 2>/dev/null | head -1)
            
            if [ -z "$preview_window" ]; then
                log_message "Preview projector not found, reopening..."
                
                # Try to reopen preview projector using OBS WebSocket or keyboard shortcuts
                # Method 1: Send F12 key to OBS main window (default preview toggle)
                local obs_main_window=$(xdotool search --name "OBS" 2>/dev/null | head -1)
                if [ -n "$obs_main_window" ]; then
                    # Focus OBS main window
                    xdotool windowfocus "$obs_main_window" 2>/dev/null
                    sleep 0.5
                    # Send F12 to toggle preview
                    xdotool key F12 2>/dev/null
                    log_message "Sent F12 to OBS to reopen preview projector"
                    sleep 2
                    
                    # Try to reposition the newly opened preview projector
                    position_preview_projector
                else
                    log_message "WARNING: OBS main window not found for preview reopening"
                fi
            else
                # Preview exists, check if it's positioned correctly on screen 3
                local window_geometry=$(xdotool getwindowgeometry "$preview_window" 2>/dev/null)
                if [ -n "$window_geometry" ]; then
                    local window_x=$(echo "$window_geometry" | grep -o "Position: [0-9]*" | cut -d' ' -f2)
                    if [ -n "$window_x" ] && [ "$window_x" -lt 3840 ]; then
                        log_message "Preview projector not on screen 3, repositioning..."
                        position_preview_projector
                    fi
                fi
            fi
            
            # Wait before next check (check every 10 seconds)
            sleep 10
        done
    ) &
    
    local monitor_pid=$!
    log_message "Preview projector monitoring started (PID: $monitor_pid)"
    return $monitor_pid
}

# Function to debug monitor detection issues
debug_monitor_detection() {
    log_message "=== Monitor Detection Debug Information ==="
    
    # Check if we're in a desktop environment
    log_message "Desktop environment check:"
    log_message "  DISPLAY: $DISPLAY"
    log_message "  XAUTHORITY: $XAUTHORITY"
    
    # Check X server status
    log_message "X server status:"
    if xset q > /dev/null 2>&1; then
        log_message "  ✓ X server is accessible"
    else
        log_message "  ✗ X server is not accessible"
    fi
    
    # Check xrandr availability
    log_message "xrandr availability:"
    if command -v xrandr > /dev/null 2>&1; then
        log_message "  ✓ xrandr is available"
        log_message "  xrandr version: $(xrandr --version | head -1)"
    else
        log_message "  ✗ xrandr is not available"
    fi
    
    # Get detailed monitor information
    log_message "Detailed monitor information:"
    if command -v xrandr > /dev/null 2>&1; then
        log_message "xrandr --listmonitors output:"
        xrandr --listmonitors 2>&1 | tee -a "$LOG_FILE"
        
        log_message "xrandr --query output:"
        xrandr --query 2>&1 | tee -a "$LOG_FILE"
        
        log_message "xrandr --listproviders output:"
        xrandr --listproviders 2>&1 | tee -a "$LOG_FILE"
    fi
    
    # Check for common monitor issues
    log_message "Common monitor issue checks:"
    
    # Check if running in Wayland
    if [ "$XDG_SESSION_TYPE" = "wayland" ]; then
        log_message "  ⚠ Running in Wayland - xrandr may not work properly"
        log_message "  Consider switching to X11 for better monitor support"
    else
        log_message "  ✓ Running in X11"
    fi
    
    # Check for NVIDIA drivers
    if command -v nvidia-smi > /dev/null 2>&1; then
        log_message "  ✓ NVIDIA drivers detected"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits 2>/dev/null | tee -a "$LOG_FILE"
    else
        log_message "  ℹ NVIDIA drivers not detected"
    fi
    
    # Check for Intel/AMD drivers
    if [ -d "/sys/class/drm" ]; then
        log_message "  ✓ DRM subsystem available"
        ls -la /sys/class/drm/ 2>/dev/null | tee -a "$LOG_FILE"
    else
        log_message "  ✗ DRM subsystem not available"
    fi
    
    # Check for common monitor tools
    log_message "Monitor management tools:"
    local tools=("xdotool" "wmctrl" "xprop" "xwininfo")
    for tool in "${tools[@]}"; do
        if command -v "$tool" > /dev/null 2>&1; then
            log_message "  ✓ $tool is available"
        else
            log_message "  ✗ $tool is not available"
        fi
    done
    
    log_message "=== End Monitor Detection Debug ==="
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
    
    # Check for window management tools (optional but recommended for preview positioning)
    if ! command -v xdotool > /dev/null 2>&1 && ! command -v wmctrl > /dev/null 2>&1; then
        log_message "WARNING: Neither xdotool nor wmctrl found - preview projector positioning may not work"
        log_message "Install with: sudo apt install xdotool wmctrl"
    else
        if command -v xdotool > /dev/null 2>&1; then
            log_message "✓ xdotool found for window positioning"
        fi
        if command -v wmctrl > /dev/null 2>&1; then
            log_message "✓ wmctrl found for window positioning"
        fi
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
    
    # Debug monitor detection issues if they occur
    log_message "Running monitor detection debug..."
    debug_monitor_detection
    
    # Setup display and multi-monitor
    if ! setup_display; then
        log_message "ERROR: Display setup failed"
        log_message "Running additional monitor debugging..."
        debug_monitor_detection
        log_message "Attempting to continue with limited monitor support..."
    else
        log_message "Display setup completed successfully"
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
    
    # Start OBS Studio with opsroom-1 scene collection
    if ! start_obs; then
        log_message "WARNING: Failed to start OBS Studio, continuing without OBS"
    else
        log_message "OBS Studio started successfully with opsroom-1 scene collection"
        
        # Position the preview projector on screen 3
        log_message "Waiting for OBS to initialize before positioning preview projector..."
        sleep 3
        if position_preview_projector; then
            log_message "Preview projector positioned on screen 3"
        else
            log_message "WARNING: Could not position preview projector on screen 3"
            log_message "This may be due to monitor detection issues or OBS not fully initialized"
        fi
        
        # Start continuous monitoring to ensure preview projector never stops
        log_message "Starting continuous preview projector monitoring..."
        if monitor_preview_projector; then
            log_message "Preview projector monitoring started successfully"
        else
            log_message "WARNING: Failed to start preview projector monitoring"
        fi
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