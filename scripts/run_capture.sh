#!/bin/bash
# Run tool_controller with refined parameters for dataset generation

# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of iterations
ITERATIONS=10

# Ellipse parameters (in meters)
RADIUS_X=0.0028       # 3mm semi-major axis
RADIUS_Y=0.0032       # 2mm semi-minor axis

# Randomization (smaller values = more consistent trajectories)
RADIUS_STD=0.0002    # 0.2mm std for radius variation
CENTER_STD=0.0002    # 0.3mm std for center offset - keeps ellipse centered

# Trajectory points
NUM_POINTS=300

# Capture settings
CAPTURE=false         # Enable/disable image capture (set false if using VIEWER)
VIEWER=true       # Enable interactive viewer (no data saving, overrides CAPTURE)
SPEED=1.0            # Animation speed for viewer mode
WIDTH=640
HEIGHT=480

# Painting settings
PAINT=true           # Enable/disable UV texture painting
PAINT_RADIUS=1       # Pixels on UV texture

# RCM (Remote Center of Motion) settings
RCM=true             # Enable/disable RCM constraint for tool orientation
RCM_RADIUS=0.005     # 5mm distance from lens center to RCM
RCM_HEIGHT=0.004     # 6mm height above lens (tool reach is 8mm)
TOOL_ROLL=-45        # Tool roll around shaft axis in degrees

# Lighting variation
VARY_LIGHTS=true     # Enable/disable per-frame lighting variation

# Debug
DEBUG=false          # Print frame idx, tool pos and euler for each frame

# =============================================================================
# BUILD COMMAND
# =============================================================================

CMD="python3 tool_controller.py --iterations $ITERATIONS"
CMD="$CMD --radius-x $RADIUS_X --radius-y $RADIUS_Y"
CMD="$CMD --radius-std $RADIUS_STD --center-std $CENTER_STD"
CMD="$CMD --num-points $NUM_POINTS"

if [ "$VIEWER" = true ]; then
    CMD="$CMD --viewer --speed $SPEED"
elif [ "$CAPTURE" = true ]; then
    CMD="$CMD --capture --width $WIDTH --height $HEIGHT"
fi

if [ "$PAINT" = true ]; then
    CMD="$CMD --paint --paint-radius $PAINT_RADIUS"
fi

if [ "$RCM" = true ]; then
    CMD="$CMD --rcm --rcm-radius $RCM_RADIUS --rcm-height $RCM_HEIGHT --tool-roll $TOOL_ROLL"
fi

if [ "$VARY_LIGHTS" = true ]; then
    CMD="$CMD --vary-lights"
fi

if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
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
echo "Viewer: $VIEWER (speed=${SPEED})"
echo "Capture: $CAPTURE (${WIDTH}x${HEIGHT})"
echo "Paint: $PAINT (radius=${PAINT_RADIUS}px)"
echo "RCM: $RCM (radius=${RCM_RADIUS}m, height=${RCM_HEIGHT}m, roll=${TOOL_ROLL}deg)"
echo "Vary lights: $VARY_LIGHTS"
echo "Debug: $DEBUG"
echo "============================================"
echo "Command: $CMD"
echo "============================================"

$CMD

echo "============================================"
echo "Complete!"
if [ "$VIEWER" = true ]; then
    echo "Viewer session ended"
elif [ "$CAPTURE" = true ]; then
    echo "Videos generated automatically (video.mp4)"
fi
echo "============================================"
