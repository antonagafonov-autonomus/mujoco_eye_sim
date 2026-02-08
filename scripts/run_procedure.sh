#!/bin/bash
# Run procedure_runner with runtime flags
# All procedure parameters are in parameters/procedures/<procedure>.json

# =============================================================================
# RUNTIME CONFIGURATION (only debug, recording, viewer flags)
# =============================================================================

# Procedure name (loads from parameters/procedures/<name>.json)
PROCEDURE="diathermy"

# Number of iterations
ITERATIONS=10

# Runtime modes
CAPTURE=true         # Enable/disable image capture
VIEWER=false        # Enable interactive viewer (no data saving, overrides CAPTURE)
SPEED=1.0            # Animation speed for viewer mode

# Debug
DEBUG=true          # Print frame idx, tool pos and euler for each frame

# =============================================================================
# BUILD COMMAND
# =============================================================================

CMD="python3 procedure_runner.py --procedure $PROCEDURE"
CMD="$CMD --iterations $ITERATIONS"

if [ "$VIEWER" = true ]; then
    CMD="$CMD --viewer --speed $SPEED"
elif [ "$CAPTURE" = true ]; then
    CMD="$CMD --capture"
fi

if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi

# =============================================================================
# RUN
# =============================================================================

echo "============================================"
echo "Running Procedure: $PROCEDURE"
echo "============================================"
echo "Iterations: $ITERATIONS"
echo "Viewer: $VIEWER (speed=${SPEED})"
echo "Capture: $CAPTURE"
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
    echo "Captures saved to ../captures/"
fi
echo "============================================"
