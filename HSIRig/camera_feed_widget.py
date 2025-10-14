"""
Details
"""
# imports
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel

# classes
class CameraFeedWidget(QLabel):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.change_pixmap_signal.connect(self.update_image)
        self.image_height = 720  # Fixed height for the image
        self.image_width = 1080  # Width based on the input line data
        self.image_data = np.zeros((self.image_height, self.image_width), dtype=np.uint8)  # Initialize with black image
        self.setPixmap(self.convert_cv_qt(self.image_data))

    def update_image(self, line_data):
        """Updates the QLabel with new line data"""
        self.add_line_to_image(line_data)
        qt_img = self.convert_cv_qt(self.image_data)
        self.setPixmap(qt_img)

    def add_line_to_image(self, line_data):
        """Append the new line data to the existing image"""
        line_data = (line_data * 255).astype(np.uint8).T  # Convert to 8-bit and transpose to horizontal

        # Shift existing lines down by one
        self.image_data = np.roll(self.image_data, shift=1, axis=0)

        # Replace the top line with the new line data
        self.image_data[0, :] = line_data.flatten()

    def reset_image_data(self):
        """Reset the image data to black"""
        self.image_data = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        self.setPixmap(self.convert_cv_qt(self.image_data))

    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QPixmap"""
        h, w = cv_img.shape
        bytes_per_line = w
        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(1080, 720, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

