"""
Details
"""
# imports
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# class
class MockCamera(QObject):
    """ Details """
    new_data_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        """ Details """
        super().__init__()
        self.running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._capture)

    def start_capture(self):
        """ Details """
        self.running = True
        self.timer.start(200)  # Capture every 200 milliseconds

    def stop_capture(self):
        """ Details """
        self.running = False
        self.timer.stop()

    def _capture(self):
        """ Details """
        if self.running:
            # Simulate line scanning data package
            data = np.random.rand(1, 1080)  # Example data, modify as needed
            self.new_data_signal.emit(data)

