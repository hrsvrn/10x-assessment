#!/bin/bash

# 1. Install project requirements
pip install -r requirements.txt

# 2. Clone SEEM Repository if not exists
if [ ! -d "Segment-Everything-Everywhere-All-At-Once" ]; then
    echo "Cloning SEEM repository..."
    git clone https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git
else
    echo "SEEM repository already exists."
fi

# 3. Install SEEM dependencies
cd Segment-Everything-Everywhere-All-At-Once
if [ -f "requirements.txt" ]; then
    echo "Installing SEEM dependencies..."
    pip install -r requirements.txt
fi

# 4. Compile CUDA operators (MultiScaleDeformableAttention) if necessary
# This is common in detection/segmentation repos like this.
# If it fails, it might be due to missing CUDA or gcc versions, but we attempt it.
if [ -f "setup.py" ]; then
     echo "Attempting to build SEEM extensions..."
     python setup.py build develop --user
fi

cd ..

echo "----------------------------------------------------------------"
echo "Setup Complete."
echo "IMPORTANT: You must add the SEEM repo to your PYTHONPATH before running scripts."
echo "Run the following command:"
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/Segment-Everything-Everywhere-All-At-Once"
echo "----------------------------------------------------------------"
