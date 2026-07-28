# Workspace setup
cd ~/Documents/Virtual-Framing
source .venv/bin/activate

# Input check
python version2.py

# Add output at /dev/video20
sudo apt install v4l2loopback-dkms v4l2loopback-utils
sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="Virtual Camera" exclusive_caps=1
ls /dev/video20

# Input + Output
python version2.py --camera 0 --virtual-camera /dev/video20

# Controls
# Fist: toggle Dynamic Bounding Box mode between 2-finger and 5-finger
# Right pinch: next effect
# Left pinch: previous effect
# Index fingers touch: toggle mirror
# C: capture active ROI with the selected effect applied
# O: toggle window and virtual camera output between clean and debug
# Q: quit

# Notes
# Existing effects run only in 2-finger mode.
# 5-finger mode draws layered translucent color slices between matching fingertips.
