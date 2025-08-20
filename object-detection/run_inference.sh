#!/bin/bash

# Wrapper script to run inference with the correct conda environment
# This ensures Grad-CAM functionality works properly

echo "Activating cnn-detection conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cnn-detection

echo "Running inference with activated environment..."
echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Check if YOLOv8_Explainer is available
python -c "import YOLOv8_Explainer; print('✓ YOLOv8_Explainer is available')" 2>/dev/null || {
    echo "✗ YOLOv8_Explainer not available in current environment"
    echo "Please ensure you're in the cnn-detection environment"
    exit 1
}

# Run the inference script with all arguments passed through
python inference.py "$@" 