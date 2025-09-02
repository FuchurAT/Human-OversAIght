#!/bin/bash

# Quick Audio Fix for Human OversAIght
# This script quickly fixes audio issues by configuring real audio hardware

echo "=== Quick Audio Fix ==="

# Check if we're in the right environment
if [ ! -f "run_multi_apps.py" ]; then
    echo "ERROR: Please run this script from the object-detection directory"
    exit 1
fi

# Set audio environment variables
export PULSE_RUNTIME_PATH="/run/user/$(id -u)/pulse"
export PULSE_COOKIE="$HOME/.config/pulse/cookie"
export PULSE_SERVER="unix:/run/user/$(id -u)/pulse/native"
export PULSE_CLIENTCONFIG="$HOME/.config/pulse/client.conf"
export PIPEWIRE_RUNTIME_DIR="/run/user/$(id -u)/pipewire-0"
export PIPEWIRE_CONFIG_DIR="$HOME/.config/pipewire"
export SDL_AUDIODRIVER="pulse"
export AUDIODEV="pulse"

echo "✓ Audio environment variables set"

# Restart PipeWire to pick up real audio hardware
echo "Restarting PipeWire to detect real audio hardware..."
systemctl --user restart pipewire pipewire-pulse
sleep 3

# Check if real audio devices are now available
echo "Checking for real audio devices..."
if command -v pw-cli > /dev/null 2>&1; then
    echo "Available audio sinks:"
    pw-cli list-objects | grep -A 10 "Audio/Sink" | grep "node.description" || echo "No real audio sinks found"
fi

# Test pygame audio directly
echo "Testing pygame audio..."
python3 -c "
import pygame
import os

# Set audio driver
os.environ['SDL_AUDIODRIVER'] = 'pulse'

try:
    pygame.mixer.init()
    print('✓ Pygame audio initialized successfully')
    
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

echo ""
echo "=== Quick Fix Summary ==="
echo "If audio is working, you can now run your application."
echo "If not, try the following:"
echo "1. Check your system volume settings"
echo "2. Try connecting headphones or external speakers"
echo "3. Run: systemctl --user restart pipewire pipewire-pulse"
echo "4. Check if your audio hardware is muted in system settings" 