#!/bin/sh

# Source - https://stackoverflow.com/questions/43028904/converting-ui-to-py-with-python-3-6-on-pyqt5
# Posted by Danilo Gasques
# Retrieved 2025-11-06, License - CC BY-SA 4.0
#
# Modified by Robert Stevenson
# Created 6 Nov 2025


python3 -m PyQt5.uic.pyuic -xd ./main_window.ui -o main_window.py
