"""
Details
"""
# imports
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow
from ui.main_window import Ui_MainWindow
from controllers.mock_camera import MockCamera
from controllers.mock_actuator import MockActuator
from controllers.scanner_controller import ScannerController
from camera_feed_widget import CameraFeedWidget
import os
from controllers.command_client import send_command
from controllers.rig_controller import RIGController



# class
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Apply stylesheets for a better visual appearance
        #self.apply_stylesheet()

        # Mock Controllers
        self.camera_controller = MockCamera()
        self.actuator_controller = MockActuator()

        # Connect signals to slots
        self.camera_controller.new_data_signal.connect(self.update_camera_feed)
        self.actuator_controller.bed_position_signal.connect(self.update_bed_position_feedback)
        self.actuator_controller.carriage_position_signal.connect(self.update_carriage_position_feedback)
        self.actuator_controller.target_position_signal.connect(self.update_target_position_feedback)
        self.bed_position_carry = None
        self.target_carry = None

        # Replace the placeholder QLabel with the CameraFeedWidget
        #self.camera_feed_widget = CameraFeedWidget(self)
        #self.horizontalLayout_2.replaceWidget(self.camera_feed, self.camera_feed_widget)
        #self.camera_feed.deleteLater()  # Remove the placeholder

        #self.configure_buttons()

        # Status tracking
        #self.status_timer = QTimer(self)
        #self.status_timer.timeout.connect(self.update_status)
        #self.status_timer.start(1000)  # Update every second

    def apply_stylesheet(self):
        # Load the stylesheet from the file
        stylesheet_file = os.path.join(os.path.dirname(__file__), '../config/qualicrop.qss')
        with open(stylesheet_file, 'r') as file:
            stylesheet = file.read()
            self.setStyleSheet(stylesheet)

    def configure_buttons(self):
        self.scan.clicked.connect(self.start_scan)
        self.stop.clicked.connect(self.stop_scan)
        self.home_bed.clicked.connect(self.home_bed_action)
        self.home_carriage.clicked.connect(self.home_carriage_action)
        self.reset.clicked.connect(self.reset_scan)

    def start_sensor(self):
        self._start_sensor_and_callback()

    def start_scan(self):
        print("main scanning")
        self.camera_controller.start_capture()
        self.actuator_controller.start_scanning()

    def stop_scan(self):
        self.camera_controller.stop_capture()
        self.actuator_controller.bed_running = False
        self.actuator_controller.carriage_running = False

    def reset_scan(self):
        self.camera_controller.stop_capture()
        self.camera_feed_widget.reset_image_data()
        self.actuator_controller.bed_running = False
        self.actuator_controller.carriage_running = False
        self.actuator_controller.home_bed()
        
    def home_bed_action(self):
        self.actuator_controller.home_bed()

    def home_carriage_action(self):
        self.actuator_controller.home_carriage()

    def update_camera_feed(self, data):
        self.camera_feed_widget.change_pixmap_signal.emit(data)

    def update_bed_position_feedback(self, position):
        self.bed_position_label.setText(f"Bed Position: {position}")
        self.bed_position_carry = position

    def update_carriage_position_feedback(self, position):
        self.carriage_position_label.setText(f"Carriage Position: {position}")

    def update_target_position_feedback(self, position):
        self.target_position_label.setText(f"Target Position: {position}")
        self.target_carry = position
    
    def update_status(self):
        # Example of updating the status; customize as needed
        if self.actuator_controller.bed_homed:
            self.bed_status_label.setText("Bed Status: Homed")
        else:
            self.bed_status_label.setText("Bed Status: Not Homed")

        if self.actuator_controller.carriage_homed:
            self.carriage_status_label.setText("Carriage Status: Homed")
        else:
            self.carriage_status_label.setText("Carriage Status: Not Homed")

        if self.bed_position_carry and self.target_carry:
            percentage = (100/self.target_carry)*self.bed_position_carry
            self.completion_percentage_label.setText(f"Completion: {round(percentage)}%")
        else:
            self.completion_percentage_label.setText("Completion: 0%")

        if self.actuator_controller.bed_running or self.actuator_controller.carriage_running:
            self.status_label.setText("Status: Running")
        elif self.actuator_controller.bed_homed and self.actuator_controller.carriage_homed:
            self.status_label.setText("Status: Homed")
        else:
            self.status_label.setText("Status: Idle")

    def btnCameraConnect_clicked(self):
        self.btnCameraConnect.setEnabled(False)
        self.btnCameraDisconnect.setEnabled(True)
        self._start_sensor_and_callback()
        #command_status,message=send_command('CONNECT')
        #if message!='OK':
        #    self.btnCameraConnect.setEnabled(True)
        #self.status_label.setText("Status: "+ str(message))

    def btnCameraDisconnect_clicked(self):
        self.btnCameraDisconnect.setEnabled(False)
        self.btnCameraConnect.setEnabled(True)
        #command_status, message = send_command('DISCONNECT')
        #if message != 'OK':
        #    self.btnCameraConnect.setEnabled(True)
        #self.status_label.setText("Status: " + str(message))

    def btnApplyAdjust_clicked(self):

        self.btnCameraConnect.setEnabled(False)
        command_status, message = send_command('CONNECT')
        command_status, message = send_command('FRAME_RATE,50')
        command_status, message = send_command('EXPOSURE_TIME,20')
        val=str(self.cmbSpectralBin.currentIndex())
        command_status, message = send_command('SPECTRAL_BINNING,'+ str(self.cmbSpectralBin.currentIndex()))
        command_status, message = send_command('SPATIAL_BINNING,' + str(self.cmbSpatialBin.currentIndex()))

        if message != 'OK':
            self.btnCameraConnect.setEnabled(True)
        self.status_label.setText("Status: " + str(message))

    def btnStartAcquire_clicked(self):
        self.btnStartAcquire.setEnabled(False)

        controller=RIGController()
        controller.connect(port="COM3")
        controller.reset_controller()
        controller.home_axes('YZ')  # axe="YZ"
        controller.set_feed_rate('4000')  # rate is mm/min
        controller.move_axis('Y', 80, rapid=False)  # move to init
        ## TODO: controller.move_axis(Y, ~, rapid=False) # White strip
        ## TODO: controller.move_axis(Y, ~, rapid=False) # Dark strip
        ## START IMAGE SCAN
        controller.set_feed_rate('443')  # rate is mm/min
        controller.move_axis('Y', 650, rapid=False)  # move whole bed under camera for scan
        controller.set_feed_rate('4000')  # rate is mm/min
        controller.move_axis('Y', 80, rapid=False)  # move back to init
        controller.disconnect()

        self.specSensor.command('Acquisition.Start')
        self.btnStopAcquire.setEnabled(True)

        ##self.specSensor.command('Acquisition.Start')
        ##self.btnStopAcquire.setEnabled(True)
        ##command_status, message = send_command('START_ACQUISITION')
        ##receive_data()
        ##tcp_listener()

        #if message != 'OK':
        #    self.btnStartAcquire.setEnabled(True)
        #self.status_label.setText("Status: " + str(message))

    def btnStopAcquire_clicked(self):
        self.btnStopAcquire.setEnabled(False)
        self.btnStartAcquire.setEnabled(True)
        self.specSensor.command('Acquisition.Stop')
        #command_status, message = send_command('STOP_ACQUISITION')
        #if message != 'OK':
        #    self.btnStopAcquire.setEnabled(True)
        #self.status_label.setText("Status: " + str(message))

    def home_bed_clicked(self):
        scanner_controller=ScannerController()
        scanner_controller.send_command('connect')
        scanner_controller.send_command('home')



