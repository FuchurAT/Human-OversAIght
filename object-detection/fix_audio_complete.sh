#!/bin/bash

# Complete Audio Fix for Human OversAIght
# This script fixes all audio issues including PipeWire, pygame, and hardware detection

echo "=== Complete Audio Fix for Human OversAIght ==="

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_message "Starting complete audio fix..."

# Check if we're in the right environment
if [ ! -f "run_multi_apps.py" ]; then
    echo "ERROR: Please run this script from the object-detection directory"
    exit 1
fi

# Step 1: Fix PipeWire configuration
log_message "Step 1: Fixing PipeWire configuration..."

# Create PipeWire configuration directory
PIPEWIRE_CONFIG_DIR="$HOME/.config/pipewire"
mkdir -p "$PIPEWIRE_CONFIG_DIR"

# Create PipeWire client configuration
PIPEWIRE_CLIENT_CONF="$PIPEWIRE_CONFIG_DIR/client.conf"
cat > "$PIPEWIRE_CLIENT_CONF" << EOF
# PipeWire client configuration for Human OversAIght

# Core settings
core.daemon = true
core.name = pipewire-0

# Link settings
link.max-buffers = 64

# Memory settings
mem.warn-mlock = false
mem.allow-mlock = true

# Clock settings
default.clock.rate = 48000
default.clock.quantum = 1024
default.clock.min-quantum = 32
default.clock.max-quantum = 2048
default.clock.quantum-limit = 8192
default.clock.quantum-floor = 4
clock.power-of-two-quantum = true

# Settings
settings.check-quantum = false
settings.check-rate = false

# Logging
log.level = 2
log.target = stderr

# CPU settings
cpu.max-align = 64

# Video settings (if needed)
default.video.width = 640
default.video.height = 480
default.video.rate.num = 25
default.video.rate.denom = 1
EOF

log_message "Created PipeWire client configuration: $PIPEWIRE_CLIENT_CONF"

# Create PulseAudio configuration directory
PULSE_CONFIG_DIR="$HOME/.config/pulse"
mkdir -p "$PULSE_CONFIG_DIR"

# Create PulseAudio client configuration
PULSE_CLIENT_CONF="$PULSE_CONFIG_DIR/client.conf"
cat > "$PULSE_CLIENT_CONF" << EOF
# PulseAudio client configuration for Human OversAIght

# Enable shared memory for better performance
enable-shm = yes

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

# Load ALSA compatibility layer
load-module module-alsa-sink
load-module module-alsa-source

# Load null sink for applications that need it
load-module module-null-sink sink_name=null sink_properties=device.description=Null_Output
EOF

log_message "Created PulseAudio client configuration: $PULSE_CLIENT_CONF"

# Step 2: Restart PipeWire services
log_message "Step 2: Restarting PipeWire services..."

# Stop existing services
systemctl --user stop pipewire pipewire-pulse 2>/dev/null
sleep 2

# Start services
systemctl --user start pipewire
sleep 3
systemctl --user start pipewire-pulse
sleep 2

# Step 3: Set environment variables
log_message "Step 3: Setting audio environment variables..."

export PULSE_RUNTIME_PATH="/run/user/$(id -u)/pulse"
export PULSE_COOKIE="$HOME/.config/pulse/cookie"
export PULSE_SERVER="unix:/run/user/$(id -u)/pulse/native"
export PULSE_CLIENTCONFIG="$HOME/.config/pulse/client.conf"
export PIPEWIRE_RUNTIME_DIR="/run/user/$(id -u)/pipewire-0"
export PIPEWIRE_CONFIG_DIR="$HOME/.config/pipewire"
export SDL_AUDIODRIVER="pulse"
export AUDIODEV="pulse"

# Create runtime directory
mkdir -p "$PULSE_RUNTIME_PATH"

log_message "✓ Audio environment variables set"

# Step 4: Check audio hardware detection
log_message "Step 4: Checking audio hardware detection..."

# Wait for PipeWire to fully start
sleep 5

# Check if real audio devices are available
if command -v pw-cli > /dev/null 2>&1; then
    log_message "Checking PipeWire audio devices..."
    pw-cli list-objects | grep -A 10 "Audio/Sink" | grep "node.description" 2>/dev/null || log_message "No real audio sinks found in PipeWire"
fi

# Check ALSA devices
if [ -f "/proc/asound/cards" ]; then
    log_message "ALSA audio cards detected:"
    cat /proc/asound/cards 2>/dev/null
fi

# Step 5: Install pygame if needed
log_message "Step 5: Checking pygame installation..."

# Try to find conda
CONDA_FOUND=false
if command -v conda > /dev/null 2>&1; then
    CONDA_FOUND=true
    log_message "Conda found in PATH"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
    log_message "Conda found at $HOME/anaconda3"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
    log_message "Conda found at $HOME/miniconda3"
fi

# Install pygame if needed
if [ "$CONDA_FOUND" = true ]; then
    log_message "Installing pygame in conda environment..."
    
    # Try to activate cnn-detection environment
    if conda env list | grep -q "cnn-detection"; then
        conda activate cnn-detection
        log_message "Activated cnn-detection environment"
        
        # Install pygame
        pip install pygame
        log_message "Installed pygame"
    else
        log_message "cnn-detection environment not found, installing pygame globally"
        pip install pygame
    fi
else
    log_message "Conda not found, trying pip install pygame"
    pip3 install pygame
fi

# Step 6: Test audio with pygame
log_message "Step 6: Testing pygame audio..."

python3 -c "
import pygame
import os
import sys

print('Testing pygame audio...')

# Set audio driver
os.environ['SDL_AUDIODRIVER'] = 'pulse'

try:
    pygame.mixer.init()
    print('✓ Pygame mixer initialized successfully')
    
    # Try to play a simple beep
    import numpy as np
    sample_rate = 44100
    duration = 0.1
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    audio_data = (audio_data * 32767).astype(np.int16)
    
    sound = pygame.sndarray.make_sound(audio_data)
    sound.play()
    pygame.time.wait(int(duration * 1000))
    print('✓ Audio playback successful')
    
    pygame.mixer.quit()
    print('✓ Audio test completed successfully')
    
except Exception as e:
    print(f'✗ Audio test failed: {e}')
    print('Trying alternative audio backends...')
    
    # Try different backends
    backends = ['alsa', 'pipewire', 'oss', 'esd']
    for backend in backends:
        try:
            os.environ['SDL_AUDIODRIVER'] = backend
            pygame.mixer.init()
            print(f'✓ Audio works with {backend} backend')
            pygame.mixer.quit()
            break
        except Exception as e2:
            print(f'✗ {backend} backend failed: {e2}')
"

# Step 7: Create environment file for future use
log_message "Step 7: Creating environment file for future use..."

ENV_FILE="$HOME/.human_oversaight_audio_env"
cat > "$ENV_FILE" << EOF
# Human OversAIght Audio Environment Variables
export PULSE_RUNTIME_PATH="/run/user/\$(id -u)/pulse"
export PULSE_COOKIE="\$HOME/.config/pulse/cookie"
export PULSE_SERVER="unix:/run/user/\$(id -u)/pulse/native"
export PULSE_CLIENTCONFIG="\$HOME/.config/pulse/client.conf"
export PIPEWIRE_RUNTIME_DIR="/run/user/\$(id -u)/pipewire-0"
export PIPEWIRE_CONFIG_DIR="\$HOME/.config/pipewire"
export SDL_AUDIODRIVER="pulse"
export AUDIODEV="pulse"

# To use these settings, run: source $ENV_FILE
EOF

log_message "Created environment file: $ENV_FILE"

# Step 8: Final status check
log_message "Step 8: Final status check..."

# Check PipeWire status
if pgrep -f "pipewire" > /dev/null; then
    log_message "✓ PipeWire is running"
else
    log_message "✗ PipeWire is not running"
fi

# Check pipewire-pulse status
if pgrep -f "pipewire-pulse" > /dev/null; then
    log_message "✓ PipeWire PulseAudio compatibility layer is running"
else
    log_message "✗ PipeWire PulseAudio compatibility layer is not running"
fi

# Check pygame installation
python3 -c "import pygame; print('✓ Pygame is installed')" 2>/dev/null || log_message "✗ Pygame is not installed"

log_message "=== Complete Audio Fix Finished ==="
log_message ""
log_message "Next steps:"
log_message "1. Try running your application: ./run_inference.sh"
log_message "2. If audio still doesn't work, run: source $ENV_FILE"
log_message "3. Check your system volume settings"
log_message "4. Try connecting headphones or external speakers"
log_message "5. If needed, restart PipeWire: systemctl --user restart pipewire pipewire-pulse" 