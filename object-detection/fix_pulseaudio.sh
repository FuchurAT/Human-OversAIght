#!/bin/bash

# PulseAudio Fix Script for Human OversAIght
# This script fixes common PulseAudio connection issues

echo "=== PulseAudio Fix Script ==="
echo "This script will fix common PulseAudio connection issues"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Do not run this script as root"
    exit 1
fi

log_message "Starting PulseAudio fix process..."

# Kill any existing PulseAudio processes
log_message "Killing existing PulseAudio processes..."
pkill -f "pulseaudio" 2>/dev/null
sleep 3

# Check if PulseAudio is installed
if ! command -v pulseaudio > /dev/null 2>&1; then
    log_message "PulseAudio not found, installing..."
    if command -v apt-get > /dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y pulseaudio pulseaudio-utils
    elif command -v yum > /dev/null 2>&1; then
        sudo yum install -y pulseaudio pulseaudio-utils
    elif command -v dnf > /dev/null 2>&1; then
        sudo dnf install -y pulseaudio pulseaudio-utils
    else
        echo "ERROR: Could not install PulseAudio automatically"
        exit 1
    fi
fi

# Create PulseAudio configuration directory
PULSE_CONFIG_DIR="$HOME/.config/pulse"
mkdir -p "$PULSE_CONFIG_DIR"

# Create client configuration
CLIENT_CONF="$PULSE_CONFIG_DIR/client.conf"
cat > "$CLIENT_CONF" << EOF
# PulseAudio client configuration for Human OversAIght

# Enable shared memory for better performance
enable-shm = yes

# Set default sink (audio output device)
# default-sink = auto

# Set default source (audio input device)
# default-source = auto

# Enable network access (for remote audio)
# load-module module-native-protocol-tcp auth-anonymous=1

# Load ALSA compatibility layer
load-module module-alsa-sink
load-module module-alsa-source

# Load null sink for applications that need it
load-module module-null-sink sink_name=null sink_properties=device.description=Null_Output

# Set reasonable buffer sizes
default-fragments = 4
default-fragment-size-msec = 25

# Enable automatic device switching
load-module module-switch-on-connect

# Enable automatic volume restoration
load-module module-stream-restore

# Enable device policy
load-module module-device-restore

# Enable card policy
load-module module-card-restore

# Enable suspend on idle
load-module module-suspend-on-idle

# Set reasonable timeouts
exit-idle-time = 20
EOF

log_message "Created client configuration: $CLIENT_CONF"

# Create daemon configuration
DAEMON_CONF="$PULSE_CONFIG_DIR/daemon.conf"
cat > "$DAEMON_CONF" << EOF
# PulseAudio daemon configuration for Human OversAIght

# Set user
user = $(whoami)

# Set runtime directory
runtime-directory = /run/user/$(id -u)/pulse

# Set cookie file
cookie-file = $HOME/.config/pulse/cookie

# Set log level
log-level = 4

# Set log target
log-target = stderr

# Enable realtime scheduling
realtime-scheduling = yes

# Set nice level
nice-level = -11

# Set exit idle time (never exit)
exit-idle-time = -1

# Set resampler
resampler = speex-float-5

# Set default sample rate
default-sample-rate = 44100

# Set alternate sample rate
alternate-sample-rate = 48000

# Set default sample format
default-sample-format = s16le

# Set default channels
default-channels = 2

# Set default channel map
default-channel-map = front-left,front-right

# Set default fragments
default-fragments = 4

# Set default fragment size
default-fragment-size-msec = 25

# Set high priority water mark
high-priority-watermark = 4194304

# Set low priority water mark
low-priority-watermark = 524288

# Set realtime priority
realtime-priority = 9

# Set exit on idle
exit-on-idle = no

# Set flat volumes
flat-volumes = yes

# Set lock memory
lock-memory = yes

# Set memory limit
memory-limit = 0

# Set CPU limit
cpu-limit = 0

# Set shm size
shm-size-bytes = 0
EOF

log_message "Created daemon configuration: $DAEMON_CONF"

# Set environment variables
export PULSE_RUNTIME_PATH="/run/user/$(id -u)/pulse"
export PULSE_COOKIE="$HOME/.config/pulse/cookie"
export PULSE_SERVER="unix:/run/user/$(id -u)/pulse/native"
export PULSE_CLIENTCONFIG="$HOME/.config/pulse/client.conf"

# Create runtime directory
mkdir -p "$PULSE_RUNTIME_PATH"

# Start PulseAudio daemon
log_message "Starting PulseAudio daemon..."
pulseaudio --daemonize --log-level=4 --log-target=stderr --realtime --no-drop-root --exit-idle-time=-1 --file="$CLIENT_CONF" --system=no

# Wait for PulseAudio to start
log_message "Waiting for PulseAudio to start..."
sleep 5

# Test PulseAudio connection
if command -v pactl > /dev/null 2>&1; then
    if pactl info > /dev/null 2>&1; then
        log_message "✓ PulseAudio connection test successful"
        
        # List available audio devices
        log_message "Available audio devices:"
        pactl list short sinks 2>/dev/null
        
        # Set default sink if available
        DEFAULT_SINK=$(pactl list short sinks | head -1 | cut -f1)
        if [ -n "$DEFAULT_SINK" ]; then
            pactl set-default-sink "$DEFAULT_SINK"
            log_message "Set default sink to: $DEFAULT_SINK"
        fi
        
    else
        log_message "✗ PulseAudio connection test failed"
        echo "Trying alternative startup method..."
        
        # Try starting without daemon
        pulseaudio --start --log-level=4 --log-target=stderr --realtime --no-drop-root --exit-idle-time=-1 --file="$CLIENT_CONF" --system=no &
        sleep 3
        
        if pactl info > /dev/null 2>&1; then
            log_message "✓ PulseAudio started with alternative method"
        else
            log_message "✗ PulseAudio still not working"
            exit 1
        fi
    fi
else
    log_message "WARNING: pactl not available for testing"
fi

# Test pygame audio
log_message "Testing pygame audio..."
python3 -c "
import pygame
pygame.mixer.init()
print('✓ Pygame mixer initialized successfully')
pygame.mixer.quit()
print('✓ Pygame mixer test completed')
" 2>/dev/null

if [ $? -eq 0 ]; then
    log_message "✓ Pygame audio test successful"
else
    log_message "✗ Pygame audio test failed"
fi

# Set environment variables for future sessions
ENV_FILE="$HOME/.pulseaudio_env"
cat > "$ENV_FILE" << EOF
# PulseAudio environment variables for Human OversAIght
export PULSE_RUNTIME_PATH="/run/user/\$(id -u)/pulse"
export PULSE_COOKIE="\$HOME/.config/pulse/cookie"
export PULSE_SERVER="unix:/run/user/\$(id -u)/pulse/native"
export PULSE_CLIENTCONFIG="\$HOME/.config/pulse/client.conf"
export SDL_AUDIODRIVER="pulse"
export AUDIODEV="pulse"
EOF

log_message "Created environment file: $ENV_FILE"
log_message "To use these settings in future sessions, run: source $ENV_FILE"

log_message "=== PulseAudio Fix Completed ==="
log_message "PulseAudio should now be working properly"
log_message "If you still have issues, try restarting your system" 