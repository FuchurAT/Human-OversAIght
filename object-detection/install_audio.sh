#!/bin/bash

# Audio installation script for Human-OversAIght
# This script installs sounddevice and necessary system dependencies

echo "Installing audio dependencies for Human-OversAIght..."

# Function to test sounddevice installation
test_sounddevice() {
    python3 -c "
import sounddevice as sd
import soundfile as sf
print(f'Sounddevice version: {sd.__version__}')
print(f'Soundfile version: {sf.__version__}')
print(f'Available devices: {len(sd.query_devices())}')
for i, device in enumerate(sd.query_devices()):
    if device['max_outputs'] > 0:
        print(f'  Device {i}: {device[\"name\"]} ({device[\"max_outputs\"]} channels)')
print('Sounddevice test successful!')
"
}

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"
    
    # Try to install system dependencies (optional for sounddevice)
    if command -v apt-get &> /dev/null; then
        echo "Attempting to install optional system dependencies..."
        sudo apt-get update
        
        # These are optional but can help with performance
        sudo apt-get install -y python3-dev || echo "Warning: Could not install python3-dev"
        sudo apt-get install -y libportaudio2 || echo "Warning: Could not install libportaudio2"
        
    elif command -v yum &> /dev/null; then
        echo "Attempting to install optional system dependencies..."
        sudo yum install -y python3-devel || echo "Warning: Could not install python3-devel"
        sudo yum install -y portaudio || echo "Warning: Could not install portaudio"
        
    elif command -v dnf &> /dev/null; then
        echo "Attempting to install optional system dependencies..."
        sudo dnf install -y python3-devel || echo "Warning: Could not install python3-devel"
        sudo dnf install -y portaudio || echo "Warning: Could not install portaudio"
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    
    if command -v brew &> /dev/null; then
        echo "Installing optional system dependencies with Homebrew..."
        brew install portaudio || echo "Warning: Could not install portaudio"
    else
        echo "Homebrew not found. Sounddevice should work without system dependencies."
    fi
    
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "Detected Windows system (MSYS/Cygwin)"
    echo "Sounddevice should work without additional system dependencies."
else
    echo "Warning: Unsupported operating system: $OSTYPE"
fi

# Install Python dependencies
echo "Installing Python dependencies..."

# Upgrade pip and setuptools first
pip3 install --upgrade pip setuptools wheel

# Try to install sounddevice and soundfile
echo "Installing sounddevice and soundfile..."
if pip3 install sounddevice soundfile; then
    echo "✓ Python audio dependencies installed successfully"
else
    echo "Warning: Could not install via requirements.txt, trying individual packages..."
    pip3 install sounddevice || echo "Error installing sounddevice"
    pip3 install soundfile || echo "Error installing soundfile"
fi

# Test sounddevice installation
echo "Testing sounddevice installation..."
if test_sounddevice; then
    echo "✓ Audio installation completed successfully!"
    echo ""
    echo "Your system now supports:"
    echo "  - Multi-channel audio output (up to 6+ channels)"
    echo "  - High-quality audio playback"
    echo "  - Real-time audio processing"
    echo "  - Ambient sound cycling"
    echo "  - Button sound effects"
    echo ""
    echo "Available audio devices:"
    python3 -c "
import sounddevice as sd
for i, device in enumerate(sd.query_devices()):
    if device['max_outputs'] > 0:
        print(f'  Device {i}: {device[\"name\"]} ({device[\"max_outputs\"]} channels)')
"
else
    echo "✗ Audio installation failed. Please check the error messages above."
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Try installing without system dependencies:"
    echo "   pip3 install sounddevice soundfile --no-deps"
    echo ""
    echo "2. If you have permission issues:"
    echo "   pip3 install --user sounddevice soundfile"
    echo ""
    echo "3. Alternative installation methods:"
    echo "   conda install -c conda-forge sounddevice soundfile"
    echo ""
    echo "4. Check if your system has audio support:"
    echo "   python3 -c 'import sounddevice; print(sounddevice.__version__)'"
fi 