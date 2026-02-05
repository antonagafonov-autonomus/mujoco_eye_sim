#!/bin/bash
# Run tool_controller with refined parameters for dataset generation

# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of iterations
ITERATIONS=10

# Ellipse parameters (in meters)
RADIUS_X=0.003       # 3mm semi-major axis
RADIUS_Y=0.003       # 2mm semi-minor axis

# Randomization (smaller values = more consistent trajectories)
RADIUS_STD=0.0002    # 0.2mm std for radius variation
CENTER_STD=0.0002    # 0.3mm std for center offset - keeps ellipse centered

# Trajectory points
NUM_POINTS=60

# Capture settings
CAPTURE=true         # Enable/disable image capture
WIDTH=640
HEIGHT=480

# Painting settings
PAINT=true           # Enable/disable UV texture painting
PAINT_RADIUS=3       # Pixels on UV texture

# RCM (Remote Center of Motion) settings
RCM=true             # Enable/disable RCM constraint for tool orientation
RCM_RADIUS=0.005     # 5mm distance from lens center to RCM
RCM_HEIGHT=0.004     # 6mm height above lens (tool reach is 8mm)

# =============================================================================
# BUILD COMMAND
# =============================================================================

CMD="python3 tool_controller.py --iterations $ITERATIONS"
CMD="$CMD --radius-x $RADIUS_X --radius-y $RADIUS_Y"
CMD="$CMD --radius-std $RADIUS_STD --center-std $CENTER_STD"
CMD="$CMD --num-points $NUM_POINTS"

if [ "$CAPTURE" = true ]; then
    CMD="$CMD --capture --width $WIDTH --height $HEIGHT"
fi

if [ "$PAINT" = true ]; then
    CMD="$CMD --paint --paint-radius $PAINT_RADIUS"
fi

if [ "$RCM" = true ]; then
    CMD="$CMD --rcm --rcm-radius $RCM_RADIUS --rcm-height $RCM_HEIGHT"
fi

# =============================================================================
# RUN
# =============================================================================

echo "============================================"
echo "Running tool_controller"
echo "============================================"
echo "Iterations: $ITERATIONS"
echo "Ellipse: ${RADIUS_X}m x ${RADIUS_Y}m"
echo "Radius std: ${RADIUS_STD}m"
echo "Center std: ${CENTER_STD}m"
echo "Points: $NUM_POINTS"
echo "Capture: $CAPTURE (${WIDTH}x${HEIGHT})"
echo "Paint: $PAINT (radius=${PAINT_RADIUS}px)"
echo "RCM: $RCM (radius=${RCM_RADIUS}m, height=${RCM_HEIGHT}m)"
echo "============================================"
echo "Command: $CMD"
echo "============================================"

$CMD

echo "============================================"
echo "Complete!"
if [ "$CAPTURE" = true ]; then
    echo "Videos generated automatically (video.mp4)"
fi
echo "============================================"
