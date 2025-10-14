"""
Details
"""
# imports
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# class
class MockActuator(QObject):
    """ Details """

    bed_position_signal = pyqtSignal(int)
    carriage_position_signal = pyqtSignal(int)
    target_position_signal = pyqtSignal(int)

    def __init__(self):
        """ Details """
        super().__init__()
        
        # bed parameters
        self.bed_position = 0
        self.bed_min = 0
        self.bed_max = 750
        self.bed_homed = False
        self.bed_running = False
        self.target_position = 750

        # carriage parameters
        self.carriage_position = 0
        self.carriage_min = 0 
        self.carriage_max = 1000
        self.carriage_homed = False
        self.carriage_running = False

        # Timers
        self.bed_timer = QTimer()
        self.bed_timer.timeout.connect(self._move_bed)
        
        self.carriage_timer = QTimer()
        self.carriage_timer.timeout.connect(self._move_carriage)

    def home_bed(self):
        """ Details """
        self.bed_homed = False
        QTimer.singleShot(3000, self._home_bed)  # 3 seconds delay for homing

    def _home_bed(self):
        """ Details """
        self.bed_position = self.bed_min
        self.bed_homed = True
        self.bed_position_signal.emit(self.bed_position)
    
    def home_carriage(self):
        """ Details """
        self.carriage_homed = False
        QTimer.singleShot(3000, self._home_carriage)  # 3 seconds delay for homing

    def _home_carriage(self):
        """ Details """
        self.carriage_position = self.carriage_min
        self.carriage_homed = True
        self.carriage_position_signal.emit(self.carriage_position)

    def start_scanning(self):
        """ Details """
        if self.bed_homed:
            print("actuator scanning")
            self.bed_target = self.target_position
            self.bed_running = True
            self.bed_timer.start(200)  # Move every 200 milliseconds
            self.target_position_signal.emit(self.target_position)
    
    def _move_bed(self):
        """ Details """
        if self.bed_running:
            if self.bed_position < self.bed_target:
                self.bed_position += 1
                self.bed_position_signal.emit(self.bed_position)
            else:
                self.bed_running = False
                self.bed_timer.stop()
    
    def move_carriage(self, target_position):
        """ Details """
        if self.carriage_homed:
            self.carriage_target = target_position
            self.carriage_running = True
            self.carriage_timer.start(200)  # Move every 200 milliseconds

    def _move_carriage(self):
        """ Details """
        if self.carriage_running:
            if self.carriage_position < self.carriage_target:
                self.carriage_position += 1
                self.carriage_position_signal.emit(self.carriage_position)
            else:
                self.carriage_running = False
                self.carriage_timer.stop()

