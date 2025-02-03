#!/bin/bash

# GUI support for WSL 2 and Docker container

# Enable X11 forwarding
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1

# Check if DISPLAY is set correctly
echo "DISPLAY is set to: $DISPLAY"

# Start container with bash shell
exec "$@"