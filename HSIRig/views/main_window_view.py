"""
Details
"""
# imports
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow
from ui.main_window import Ui_MainWindow
from controllers.mock_camera import MockCamera
from controllers.mock_actuator import MockActuator
from controllers.rig_controller import RIGController
from camera_feed_widget import CameraFeedWidget
import os
import time
from controllers.command_client import send_command
from controllers.rig_controller import RIGController


class ScanWorkerThread(QThread):
    # signal that enable comms back to gui main thread
    status_update = pyqtSignal(str)
    scan_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, main_window, rig_controller, cam_height=0.0, scan_speed=443.33, 
                 init_pos=80.0, scan_pos=650.0):
        super().__init__()
        self.main_window = main_window
        self.rig_controller = rig_controller
        self.cam_height = cam_height
        self.scan_speed = scan_speed
        self.init_pos = init_pos
        self.scan_pos = scan_pos
        self._is_running = True
    
    def stop(self):
        """Stop the scan routine"""
        self._is_running = False

    def run(self):
        """This runs in a separate thread"""
        try:
            TIMEOUT = 95
            TRAVEL_SPEED = 6000
            
            self.status_update.emit("Starting scan routine...")
            
            if not self.rig_controller.serial_conn or not self.rig_controller.serial_conn.is_open:
                self.error_occurred.emit("ERROR: Not connected to controller!")
                return
                
            # Step 1: Reset and home
            self.status_update.emit("Resetting controller and homing axes...")
            self.rig_controller.reset_controller()
            time.sleep(2)
            
            response = self.rig_controller.home_axes()
            
            # Wait for homing with interruptible sleep
            self.status_update.emit("Waiting for homing to complete...")
            start_time = time.time()
            while time.time() - start_time < TIMEOUT and self._is_running:
                pos = self.rig_controller.get_current_position()
                if pos is not None and pos["Y"] == 0.0 and pos["Z"] == 0.0:
                    break
                self.msleep(100)  # Use QThread's msleep for better integration
                
            if not self._is_running:
                return
                
            # Move to initial position
            self.status_update.emit("Moving to initial position...")
            self.rig_controller.set_feed_rate(TRAVEL_SPEED)
            self.rig_controller.move_axis("Y", self.init_pos)
            
            start_time = time.time()
            while time.time() - start_time < TIMEOUT and self._is_running:
                pos = self.rig_controller.get_current_position()
                if pos is not None and pos["Y"] == self.init_pos and pos["Z"] == self.cam_height:
                    break
                self.msleep(100)
                
            if not self._is_running:
                return
                
            # Start acquisition
            self.status_update.emit("Starting camera acquisition...")
            self.msleep(2000)
            self.main_window.specSensor.command('Acquisition.Start')
            
            # Start scan
            self.status_update.emit(f"Scanning to position {self.scan_pos}...")
            self.rig_controller.set_feed_rate(self.scan_speed)
            self.rig_controller.move_axis("Y", self.scan_pos)
            
            # Wait for scan completion
            start_time = time.time()
            while time.time() - start_time < TIMEOUT and self._is_running:
                pos = self.rig_controller.get_current_position()
                if pos is not None and pos["Y"] == self.scan_pos and pos["Z"] == self.cam_height:
                    break
                self.msleep(100)
                
            if not self._is_running:
                return
                
            # Return to initial position
            self.status_update.emit("Returning to initial position...")
            self.rig_controller.set_feed_rate(TRAVEL_SPEED)
            self.rig_controller.move_axis("Y", self.init_pos)
            
            start_time = time.time()
            while time.time() - start_time < TIMEOUT and self._is_running:
                pos = self.rig_controller.get_current_position()
                if pos is not None and pos["Y"] == self.init_pos and pos["Z"] == self.cam_height:
                    break
                self.msleep(100)
                
            self.status_update.emit("Scan routine completed successfully!")
            self.scan_complete.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Error during scan: {str(e)}")

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

    def run_scan_routine(self, rig_controller=None, cam_height=0.0, scan_speed=443.33, init_pos=80.0, scan_pos=650.0,
                         camera_capture_function=None):
        """
        Run a test routine for scanning:
        1. Reset controller and home axes
        2. Move Y axis to position 80 at rate 4000 mm/min
        3. Wait until position is reached
        4. Set rate to 443 mm/min and move to position 650
        5. Move back to position 80 at rate 4000 mm/min

        Parameters:
        - custom_function: A function to run at a specific point during the scan routine (e.g., camera capture).
        """
        if rig_controller is None:
            return False

        TIMEOUT = 95
        TRAVEL_SPEED = 6000

        print("\n=== Starting Test Routine ===\n")

        if not rig_controller.serial_conn or not rig_controller.serial_conn.is_open:
            print("ERROR: Not connected to controller!")
            return False

        # Step 1: Reset and home
        print("Step 1: Resetting controller and homing axes...")
        rig_controller.reset_controller()
        time.sleep(2)

        # Home without user confirmation
        print("Homing Y and Z axes...")
        response = rig_controller.home_axes()
        print(f"Homing response: {response}")

        # Wait for homing to complete by checking status
        print("Waiting for homing to complete...")
        start_time = time.time()
        while time.time() - start_time < TIMEOUT:  # HOMING takes a max time of 90 secs (95s is safe)
            pos = rig_controller.get_current_position()
            if pos is not None and pos["Y"] == 0.0 and pos["Z"] == 0.0:
                print(f"Homing finished early @ \n{pos}")
                break  # Condition met
            time.sleep(0.1)  # Check every 100ms
        print("Homing finished")

        # Step 3: Move and wait till in camera scan init position
        print("Moving into initial pos")
        rig_controller.set_feed_rate(TRAVEL_SPEED)
        rig_controller.move_axis("Y", init_pos)

        # Wait until we are at the initial position
        start_time = time.time()
        while time.time() - start_time < TIMEOUT:
            pos = rig_controller.get_current_position()
            if pos is not None and pos["Y"] == init_pos and pos["Z"] == cam_height:
                print("At initial position early")
                break  # Condition met
            time.sleep(0.1)  # Check every 100ms
        print("At initial position")

        # Step 4: Set rate to 443 and move to 650
        print("\nStep 4: Moving to Y position 650mm at 443mm/min...")
        rig_controller.set_feed_rate(scan_speed)

        # If a custom function is provided, call it here (e.g., start camera capture)
        # if camera_capture_function:
        #     print(f"!!! Running Camera Capture Function: {camera_capture_function.__name__} !!!")
            # camera_capture_function()  # Call the custom function

        # START Camera Acquire actions
        time.sleep(2)
        self.specSensor.command('Acquisition.Start'),
        self.btnStopAcquire.setEnabled(True)
        # we wait a fit for it to start up
        start_time = time.time()
        while time.time() - start_time < 2:
            time.sleep(0.1)  # Check every 100ms

        # Move to the scan position
        rig_controller.move_axis("Y", scan_pos)
        # Wait until we are at the scan position
        start_time = time.time()
        while time.time() - start_time < TIMEOUT:
            pos = rig_controller.get_current_position()
            if pos is not None and pos["Y"] == scan_pos and pos["Z"] == cam_height:
                print("Finished Full bed scan early")
                break  # Condition met
            time.sleep(0.1)  # Check every 100ms
        print("Full bed scan finished")

        # Step 5: Move back to Y position 80mm at 4000mm/min
        print("\nStep 5: Moving back to Y position 80mm at 4000mm/min...")
        rig_controller.set_feed_rate(TRAVEL_SPEED)
        rig_controller.move_axis("Y", init_pos)

        # Wait until we are back at the initial position
        start_time = time.time()
        while time.time() - start_time < TIMEOUT:
            pos = rig_controller.get_current_position()
            if pos is not None and pos["Y"] == init_pos and pos["Z"] == cam_height:
                print("Back to initial position early")
                break  # Condition met
            time.sleep(0.1)  # Check every 100ms

        print("\n=== Test Routine Completed Successfully ===\n")
        return True

    def btnStartAcquire_clicked(self):
        self.btnStartAcquire.setEnabled(False)
        self.btnStopAcquire.setEnabled(True)
        
        # Create and connect the RIG controller
        controller = RIGController()
        controller.connect(port="COM3")
        
        # Create worker thread
        self.scan_worker = ScanWorkerThread(
            main_window=self,
            rig_controller=controller
        )
        
        # Connect signals
        self.scan_worker.status_update.connect(self.on_scan_status_update)
        self.scan_worker.scan_complete.connect(self.on_scan_complete)
        self.scan_worker.error_occurred.connect(self.on_scan_error)
        
        # Start the thread
        self.scan_worker.start()


    #   Pre-Async/Thread:
        # self.btnStartAcquire.setEnabled(False)

        # controller=RIGController()
        # controller.connect(port="COM3") #TODO: this port should be selected via a dropdown box

        # capture_lambda = lambda: (self.specSensor.command('Acquisition.Start'),
        #                      self.btnStopAcquire.setEnabled(True))
        # self.run_scan_routine(
        #     rig_controller=controller,
        #     camera_capture_function=capture_lambda
        # )
        # controller.disconnect()

# Old? :
        ##self.specSensor.command('Acquisition.Start')
        ##self.btnStopAcquire.setEnabled(True)
        ##command_status, message = send_command('START_ACQUISITION')
        ##receive_data()
        ##tcp_listener()

        #if message != 'OK':
        #    self.btnStartAcquire.setEnabled(True)
        #self.status_label.setText("Status: " + str(message))
    def on_scan_status_update(self, message):
        """Update status label with scan progress"""
        self.status_label.setText(f"Status: {message}")
        
    def on_scan_complete(self):
        """Handle scan completion"""
        self.btnStartAcquire.setEnabled(True)
        self.btnStopAcquire.setEnabled(False)
        self.specSensor.command('Acquisition.Stop')
        self.status_label.setText("Status: Scan completed")
        
        # Disconnect controller if it exists
        if hasattr(self, 'scan_worker') and self.scan_worker.rig_controller:
            self.scan_worker.rig_controller.disconnect()
            
    def on_scan_error(self, error_message):
        """Handle scan errors"""
        self.btnStartAcquire.setEnabled(True)
        self.btnStopAcquire.setEnabled(False)
        self.status_label.setText(f"Error: {error_message}")
        
        # Try to stop acquisition
        try:
            self.specSensor.command('Acquisition.Stop')
        except:
            pass
    def btnStopAcquire_clicked(self):
        self.btnStopAcquire.setEnabled(False)
        self.btnStartAcquire.setEnabled(True)
        
        # Stop the worker thread if it exists
        if hasattr(self, 'scan_worker') and self.scan_worker.isRunning():
            self.scan_worker.stop()
            self.scan_worker.wait(2000)  # Wait up to 2 seconds for thread to finish
            
        # Stop acquisition
        self.specSensor.command('Acquisition.Stop')
        self.status_label.setText("Status: Stopped")

    #   Pre-Async/Thread:
        # self.btnStopAcquire.setEnabled(False)
        # self.btnStartAcquire.setEnabled(True)
        # self.specSensor.command('Acquisition.Stop')
        #command_status, message = send_command('STOP_ACQUISITION')
        #if message != 'OK':
        #    self.btnStopAcquire.setEnabled(True)
        #self.status_label.setText("Status: " + str(message))

    def home_bed_clicked(self):
        scanner_controller=RIGController()
        scanner_controller.send_command('connect')
        scanner_controller.send_command('home')



