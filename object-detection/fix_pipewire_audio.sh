#!/bin/bash

# PipeWire Audio Fix Script for Human OversAIght
# This script fixes audio issues on systems using PipeWire (modern replacement for PulseAudio)

echo "=== PipeWire Audio Fix Script ==="
echo "This script will fix audio issues on PipeWire systems"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Do not run this script as root"
    exit 1
fi

log_message "Starting PipeWire audio fix process..."

# Check if PipeWire is running
if ! pgrep -f "pipewire" > /dev/null; then
    log_message "ERROR: PipeWire is not running"
    log_message "Starting PipeWire..."
    systemctl --user start pipewire pipewire-pulse
    sleep 3
fi

# Check PipeWire status
if pgrep -f "pipewire" > /dev/null; then
    log_message "✓ PipeWire is running"
else
    log_message "✗ PipeWire failed to start"
    exit 1
fi

# Check if pipewire-pulse is running (PulseAudio compatibility layer)
if pgrep -f "pipewire-pulse" > /dev/null; then
    log_message "✓ PipeWire PulseAudio compatibility layer is running"
else
    log_message "Starting PipeWire PulseAudio compatibility layer..."
    systemctl --user start pipewire-pulse
    sleep 2
fi

# Set environment variables for PipeWire
export PULSE_RUNTIME_PATH="/run/user/$(id -u)/pulse"
export PULSE_COOKIE="$HOME/.config/pulse/cookie"
export PULSE_SERVER="unix:/run/user/$(id -u)/pulse/native"
export PULSE_CLIENTCONFIG="$HOME/.config/pulse/client.conf"

# Create runtime directory if it doesn't exist
mkdir -p "$PULSE_RUNTIME_PATH"

# Create PipeWire client configuration
PULSE_CONFIG_DIR="$HOME/.config/pulse"
mkdir -p "$PULSE_CONFIG_DIR"

# Create client configuration for PipeWire compatibility
CLIENT_CONF="$PULSE_CONFIG_DIR/client.conf"
cat > "$CLIENT_CONF" << EOF
# PipeWire client configuration for Human OversAIght
# This configuration ensures compatibility with pygame and other audio applications

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

# PipeWire specific settings
# Enable PipeWire compatibility
load-module module-pipewire-sink
load-module module-pipewire-source
EOF

log_message "Created PipeWire client configuration: $CLIENT_CONF"

# Test PipeWire connection
log_message "Testing PipeWire connection..."
if command -v pw-cli > /dev/null 2>&1; then
    if pw-cli info all > /dev/null 2>&1; then
        log_message "✓ PipeWire connection test successful"
        
        # List available audio devices
        log_message "Available audio devices:"
        pw-cli list-objects | grep -A 5 -B 5 "Audio/Sink" 2>/dev/null || echo "No audio sinks found"
        
    else
        log_message "✗ PipeWire connection test failed"
    fi
else
    log_message "WARNING: pw-cli not available for testing"
fi

# Test PulseAudio compatibility layer
log_message "Testing PulseAudio compatibility layer..."
if command -v pactl > /dev/null 2>&1; then
    if pactl info > /dev/null 2>&1; then
        log_message "✓ PulseAudio compatibility layer working"
        
        # List available sinks
        log_message "Available PulseAudio sinks:"
        pactl list short sinks 2>/dev/null || echo "No PulseAudio sinks found"
        
    else
        log_message "✗ PulseAudio compatibility layer not working"
    fi
else
    log_message "WARNING: pactl not available for testing"
fi

# Check for real audio hardware
log_message "Checking for audio hardware..."
if [ -d "/proc/asound" ]; then
    log_message "ALSA subsystem available"
    if [ -f "/proc/asound/cards" ]; then
        log_message "ALSA cards:"
        cat /proc/asound/cards 2>/dev/null || echo "No ALSA cards found"
    fi
else
    log_message "WARNING: ALSA subsystem not available"
fi

# Test pygame audio with different backends
log_message "Testing pygame audio with different backends..."

# Test 1: Default pygame mixer
python3 -c "
import pygame
try:
    pygame.mixer.init()
    print('✓ Pygame mixer initialized successfully with default settings')
    pygame.mixer.quit()
except Exception as e:
    print(f'✗ Default pygame mixer failed: {e}')
" 2>/dev/null

# Test 2: Pygame with PulseAudio backend
python3 -c "
import pygame
import os
os.environ['SDL_AUDIODRIVER'] = 'pulse'
try:
    pygame.mixer.init()
    print('✓ Pygame mixer initialized successfully with PulseAudio backend')
    pygame.mixer.quit()
except Exception as e:
    print(f'✗ PulseAudio pygame mixer failed: {e}')
" 2>/dev/null

# Test 3: Pygame with ALSA backend
python3 -c "
import pygame
import os
os.environ['SDL_AUDIODRIVER'] = 'alsa'
try:
    pygame.mixer.init()
    print('✓ Pygame mixer initialized successfully with ALSA backend')
    pygame.mixer.quit()
except Exception as e:
    print(f'✗ ALSA pygame mixer failed: {e}')
" 2>/dev/null

# Test 4: Pygame with PipeWire backend
python3 -c "
import pygame
import os
os.environ['SDL_AUDIODRIVER'] = 'pipewire'
try:
    pygame.mixer.init()
    print('✓ Pygame mixer initialized successfully with PipeWire backend')
    pygame.mixer.quit()
except Exception as e:
    print(f'✗ PipeWire pygame mixer failed: {e}')
" 2>/dev/null

# Create environment file for future sessions
ENV_FILE="$HOME/.pipewire_audio_env"
cat > "$ENV_FILE" << EOF
# PipeWire audio environment variables for Human OversAIght
export PULSE_RUNTIME_PATH="/run/user/\$(id -u)/pulse"
export PULSE_COOKIE="\$HOME/.config/pulse/cookie"
export PULSE_SERVER="unix:/run/user/\$(id -u)/pulse/native"
export PULSE_CLIENTCONFIG="\$HOME/.config/pulse/client.conf"

# Pygame audio backend settings
export SDL_AUDIODRIVER="pulse"
export AUDIODEV="pulse"

# PipeWire specific settings
export PIPEWIRE_RUNTIME_DIR="/run/user/\$(id -u)/pipewire-0"
export PIPEWIRE_CONFIG_DIR="\$HOME/.config/pipewire"
EOF

log_message "Created environment file: $ENV_FILE"
log_message "To use these settings in future sessions, run: source $ENV_FILE"

# Create a simple audio test script
AUDIO_TEST_SCRIPT="$HOME/test_audio.py"
cat > "$AUDIO_TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Simple audio test script for Human OversAIght
"""

import pygame
import os
import sys

def test_audio_backend(backend_name, env_var=None):
    """Test pygame audio with a specific backend"""
    print(f"Testing {backend_name} backend...")
    
    # Set environment variable if provided
    if env_var:
        os.environ['SDL_AUDIODRIVER'] = env_var
    
    try:
        pygame.mixer.init()
        print(f"✓ {backend_name} backend: SUCCESS")
        
        # Try to play a simple beep
        try:
            # Create a simple sine wave
            import numpy as np
            sample_rate = 44100
            duration = 0.1  # 100ms
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Create a pygame Sound object
            sound = pygame.sndarray.make_sound(audio_data)
            sound.play()
            pygame.time.wait(int(duration * 1000))
            print(f"✓ {backend_name} backend: Audio playback successful")
            
        except Exception as e:
            print(f"⚠ {backend_name} backend: Audio playback failed: {e}")
        
        pygame.mixer.quit()
        return True
        
    except Exception as e:
        print(f"✗ {backend_name} backend: FAILED - {e}")
        return False

def main():
    print("=== Audio Backend Test ===")
    
    # Test different backends
    backends = [
        ("Default", None),
        ("PulseAudio", "pulse"),
        ("ALSA", "alsa"),
        ("PipeWire", "pipewire"),
        ("OSS", "oss"),
        ("ESD", "esd"),
        ("NAS", "nas"),
        ("WASAPI", "wasapi"),
        ("DirectSound", "directsound"),
    ]
    
    successful_backends = []
    
    for backend_name, env_var in backends:
        if test_audio_backend(backend_name, env_var):
            successful_backends.append((backend_name, env_var))
        print()
    
    print("=== Test Results ===")
    if successful_backends:
        print("Successful backends:")
        for backend_name, env_var in successful_backends:
            print(f"  ✓ {backend_name}")
        
        # Recommend the best backend
        recommended = None
        for backend_name, env_var in successful_backends:
            if backend_name in ["PulseAudio", "PipeWire"]:
                recommended = (backend_name, env_var)
                break
        
        if not recommended and successful_backends:
            recommended = successful_backends[0]
        
        if recommended:
            print(f"\nRecommended backend: {recommended[0]}")
            if recommended[1]:
                print(f"Set SDL_AUDIODRIVER={recommended[1]} for best compatibility")
    else:
        print("No audio backends worked!")
        print("This indicates a system-level audio configuration issue.")
    
    return len(successful_backends) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x "$AUDIO_TEST_SCRIPT"
log_message "Created audio test script: $AUDIO_TEST_SCRIPT"

# Run the audio test
log_message "Running comprehensive audio test..."
python3 "$AUDIO_TEST_SCRIPT"

log_message "=== PipeWire Audio Fix Completed ==="
log_message "If audio still doesn't work, try the following:"
log_message "1. Run the audio test script: python3 $AUDIO_TEST_SCRIPT"
log_message "2. Check if your audio hardware is properly detected"
log_message "3. Try restarting PipeWire: systemctl --user restart pipewire pipewire-pulse"
log_message "4. Check audio volume and mute settings"
log_message "5. Try connecting headphones or external speakers" 