"""
Details
"""
# imports
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QMutex, QWaitCondition, QMutexLocker, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from ui.main_window import Ui_MainWindow
from controllers.rig_controller import RIGController
import os
import time
from controllers.command_client import send_command
from controllers.rig_controller import RIGController
import rig_settings

import ctypes as C
import numpy as np
from queue import Queue, Empty

# Matplotlib canvas for Qt camera preview
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from SpecSensor import SpecSensor

WIDTH = 1024
BAND_IDXS = np.array([15, 60, 90], dtype=int)  # R,G,B (0-based)
ROLLING_HEIGHT = 512
QUEUE_MAX = 4096

# class
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # Track what tab we are on
        self.current_tab = 0
        # track if camera is connected (NOTE: Might be better way to get this from SDK request)
        self.cam_connected = False

        self.add_camera_preview_ui()

        # rig controller object that should be used when connecting and commanding the rig
        self.rig_controller: RIGController = None 
        # Setup the `setup` tab in the UI for the Rig Control and connections
        self.setup_rig_ui()

    def add_camera_preview_ui(self):
        # --- replace QLabel with Matplotlib canvas ---
        self.rgb_img = np.zeros((ROLLING_HEIGHT, WIDTH, 3), dtype=np.uint8)

        self.canvas = FigureCanvas(Figure(figsize=(6, 3)))
        self.ax = self.canvas.figure.subplots()
        self.ax.axis("off")
        self.im = self.ax.imshow(self.rgb_img, vmin=0, vmax=255,
                                 aspect="auto", interpolation="nearest")

        # swap widgets in the same layout spot
        self.horizontalLayout_2.replaceWidget(self.camera_feed, self.canvas)
        self.camera_feed.deleteLater()

        # queue between SDK callback (producer) and UI (consumer)
        self.line_queue = Queue(maxsize=QUEUE_MAX)

        # stats
        self._lines_rcvd = 0
        self._bands_in_line = None

        # start camera and register callback
        #self._start_sensor_and_callback()

        # UI timer to update plot (main thread only)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._drain_and_update_plot)
        self.timer.start(30)  # ~33 FPS UI
        # --- end of eplace QLabel with Matplotlib canvas ---

    def setup_rig_ui(self):
        # setup the buttons for the rig control/settinga page
        self.configure_buttons()

        # get the available com ports and add them to combobox list
        self.update_comm_port_list()
        # lets also udate the list everytime we change to the setup tab
        self.tabWidget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        # tab 1 (index=1) is the Setup tab for the Rig
        self.current_tab = index # update the current tab index we are on (NOTE: there is probably a better pyQT way of getting this info)
        if index == 1:
            self.update_comm_port_list()

    def update_comm_port_list(self):
        """Refresh the available COM ports in the dropdown"""
        self.cmbBoxCommPortSelect.clear()  # Clear previous items
        self.cmbBoxCommPortSelect_2.clear()  # Clear previous items

        # get the list of ports using the rig controller
        ports = RIGController.list_ports(self=RIGController) 
        if not ports:
            self.cmbBoxCommPortSelect.addItem("No ports found")
            self.cmbBoxCommPortSelect_2.addItem("No ports found")
            self.cmbBoxCommPortSelect.setEnabled(False)
            self.cmbBoxCommPortSelect_2.setEnabled(False)
        else:
            for port in ports:
                display_text = f"{port.device} ({port.description})"
                self.cmbBoxCommPortSelect.addItem(display_text, port.device)
                self.cmbBoxCommPortSelect_2.addItem(display_text, port.device)
            self.cmbBoxCommPortSelect.setEnabled(True)
            self.cmbBoxCommPortSelect_2.setEnabled(True)

    def apply_stylesheet(self):
        # Load the stylesheet from the file
        stylesheet_file = os.path.join(os.path.dirname(__file__), '../config/qualicrop.qss')
        with open(stylesheet_file, 'r') as file:
            stylesheet = file.read()
            self.setStyleSheet(stylesheet)

    def configure_buttons(self):
        # Rig Connect/Disconnect buttons configure
        self.btnRigConnect.clicked.connect(self.connect_controller)
        self.btnRigDisconnect.clicked.connect(self.disconnect_controller)

        # Rig update settings button link
        self.btnRigUpdateSettings.clicked.connect(self.update_rig_settings)

        # Rig controller buttons configure
        self.btnHomeBed.clicked.connect(self.home_bed_clicked)
        self.btnHomeCarriage.clicked.connect(self.home_carriage_clicked)
        self.btnWhiteStrip.clicked.connect(self.move_to_white_calibration)
        self.btnBlackStrip.clicked.connect(self.move_to_black_calibration)
        # self.btnScan.clicked.connect(self.start_scan)
        # self.btnStop.clicked.connect(self.stop)
        self.btnReset.clicked.connect(self.reset_controller)
    
    # saving rig controller scanning config to globally accessable python config file
    def update_rig_settings(self):
        def update_txt_value(var, txtbox):
            '''update the variable if the txtbox has values and return it is so, else keep value the same'''
            value = txtbox.text()
            if str(value) != "":
                var = float(value) 
            return var
        
        # Reassign the updated value back to rig_settings.RIG_SPEED, since Python passes floats by value (immutable);
        # updating inside the function doesn't affect the original variable.
        rig_settings.RIG_SPEED = update_txt_value(rig_settings.RIG_SPEED, self.txtSpeed)
        rig_settings.RIG_BED_START = update_txt_value(rig_settings.RIG_BED_START, self.txtBedStartPosition)
        rig_settings.RIG_BED_END = update_txt_value(rig_settings.RIG_BED_END, self.txtBedEndPosition)
        rig_settings.RIG_CAM_HEIGHT = update_txt_value(rig_settings.RIG_CAM_HEIGHT, self.txtCameraPosition)

        #debug prints
        print(f"rig speed: {rig_settings.RIG_SPEED}")
        print(f"rig bed start: {rig_settings.RIG_BED_START}")
        print(f"rig bed end: {rig_settings.RIG_BED_END}")
        print(f"rig cam height: {rig_settings.RIG_CAM_HEIGHT}")

    # connecting to selected COM port from dropdown list
    def connect_controller(self):
        if self.current_tab == 0: # if we are on the camera connection tab
            selected_index = self.cmbBoxCommPortSelect_2.currentIndex()
            port_info = self.cmbBoxCommPortSelect_2.itemData(selected_index) 
        else: # else we are on the rig control tab
            selected_index = self.cmbBoxCommPortSelect.currentIndex()
            port_info = self.cmbBoxCommPortSelect.itemData(selected_index)
        if not port_info:
            print("Status: No port selected")
            return

        self.rig_controller = RIGController(port=port_info)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        if self.rig_controller.connect():
            print(f"Status: Connected to {port_info}")
            self.btnRigConnect.setEnabled(False)
            self.btnRigDisconnect.setEnabled(True)

            self.btnRigConnect_2.setEnabled(False)
            self.btnRigDisconnect_2.setEnabled(True)
        else:
            print("Status: Failed to connect")
            print(f"Rig Controller: {self.rig_controller}")
        QApplication.restoreOverrideCursor()

    # disconnecting from controller
    def disconnect_controller(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        if self.rig_controller != None and self.rig_controller.is_connected():
            self.rig_controller.disconnect()
            self.rig_controller = None # clear reference to object after disconnect
            print("Status: Disconnected")
            self.btnRigConnect.setEnabled(True)
            self.btnRigDisconnect.setEnabled(False)

            self.btnRigConnect_2.setEnabled(True)
            self.btnRigDisconnect_2.setEnabled(False)
        else:
            print("Status: Not connected")
        QApplication.restoreOverrideCursor()


    # Rig movement to white calibration stip
    def move_to_white_calibration(self):
        print(f"Moving to white calibration strip")
        if self.rig_controller != None and not self.rig_controller.is_connected():
            print("Status: Bed controller not connected")
            return

        # NOTE: You may need to update the Y/Z positions for your strip in rig_settings.py if incorrect
        y_pos = rig_settings.RIG_WHITE_CAL_POS_READ_ONLY # Position still needs calibrating
        self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)  # Set feedrate in mm/min

        # Move to Y axis position (strip location)
        success_y = self.rig_controller.move_axis('Y', y_pos)
        
        # TODO: Move this homing to seperate thread process so that it is not blocking
        # Wait for moving with interruptible sleep
        print("Waiting for move to white calibration strip to complete...")
        start_time = time.time()
        while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY:
            pos = self.rig_controller.get_current_position()
            if pos is not None and pos["Y"] == rig_settings.RIG_WHITE_CAL_POS_READ_ONLY and pos["Z"] == rig_settings.RIG_CAM_HEIGHT:
                break
            try:
                self.msleep(100)  # Use QThread's msleep for better integration
            except:
                time.sleep(0.1)
            
        if success_y:
            print("Status: At white calibration strip")
        else:
            print("Status: Move failed")

    # Rig movement to black calibration stip
    def move_to_black_calibration(self):
        print(f"Moving to black calibration strip")
        if self.rig_controller != None and not self.rig_controller.is_connected():
            print("Status: Bed controller not connected")
            return

        # NOTE: You may need to update the Y/Z positions for your strip in rig_settings.py if incorrect
        y_pos = rig_settings.RIG_BLACK_CAL_POS_READ_ONLY # Position still needs calibrating
        self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)  # Set feedrate in mm/min

        # Move to Y axis position (strip location)
        success_y = self.rig_controller.move_axis('Y', y_pos)
        
        # TODO: Move this homing to seperate thread process so that it is not blocking
        # Wait for moving with interruptible sleep
        print("Waiting for move to black calibration strip to complete...")
        start_time = time.time()
        while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY:
            pos = self.rig_controller.get_current_position()
            if pos is not None and pos["Y"] == rig_settings.RIG_BLACK_CAL_POS_READ_ONLY and pos["Z"] == rig_settings.RIG_CAM_HEIGHT:
                break
            try:
                self.msleep(100)  # Use QThread's msleep for better integration
            except:
                time.sleep(0.1)
            
        if success_y:
            print("Status: At black calibration strip")
        else:
            print("Status: Move failed")

    # Rig homing of bed
    def home_bed_clicked(self):
        if self.rig_controller != None and not self.rig_controller.is_connected():
            print("Status: controller not connected")
            return

        print("Status: Homing Y")
        try:
            success = self.rig_controller.home_axes(axes='Y')
            if success:
                print("Bed Status: Homed")
                print("Status: Bed homed")
            else:
                print("Status: Homing failed")
        finally:
            pass
    
    # Rig homing of carriage (camera)
    def home_carriage_clicked(self):
        if self.rig_controller != None and not self.rig_controller.is_connected():
            print("Status: controller not connected")
            return

        print("Status: Homing Z")
        try:
            success = self.rig_controller.home_axes(axes='Z')
            if success:
                print("Bed Status: Homed")
                print("Status: carriage homed")
            else:
                print("Status: Homing failed")
        finally:
            pass
        
    # rig controller reset logic
    def reset_controller(self):
        if not self.rigcontroller.is_connected():
            print("Controller not connected")
            return

        print("Resetting controller...")
        response = self.rigcontroller.reset_controller()
        if response:
            print("Reset response:", response)
        else:
            print("Reset unsuccessful")

    # ===================
    # === CAMERA CODE ===
    # ===================

    def _start_sensor_and_callback(self):
        self.specSensor = SpecSensor(sdkFolder='./libs/')
        profiles = self.specSensor.getProfiles()
        if not profiles:
            QtWidgets.QMessageBox.critical(self, "SpecSensor", "No devices found.")
            return

        # open first profile (or pick by name)
        # SELECT CAMERA VIA THE INDEX (FX10e == 15)
        err, _ = self.specSensor.open(profiles[15], autoInit=True)
        if err != 0:
            QtWidgets.QMessageBox.critical(self, "SpecSensor", f"Open failed: {err}")
            self.cam_connected = False
            return

        # Keep strong reference so it won't be GC’ed
        self._callback_ref = self._onDataCallback
        self.specSensor.sensor.registerDataCallback(self._callback_ref)
        
        # track success connection
        self.cam_connected = True

        # start acquisition
        #self.specSensor.command('Acquisition.Start')

    def _reshape_line_from_bytes(self, pBuffer, nbytes):
        if nbytes % 2:
            return None, None
        nsamples = nbytes // 2
        u16_ptr = C.cast(pBuffer, C.POINTER(C.c_uint16))
        raw = np.ctypeslib.as_array(u16_ptr, shape=(nsamples,))

        if nsamples % WIDTH != 0:
            return None, None
        bands = nsamples // WIDTH

        # BIL for a single line -> (bands, width).T == (pixels, bands)
        line16 = raw.reshape(bands, WIDTH).T
        return line16, bands

    def _u16_to_u8(self, x16, eps=1e-6):
        # robust 1–99% percentile stretch per channel
        lo = np.percentile(x16, 1, axis=0)
        hi = np.percentile(x16, 99, axis=0)
        y = (x16 - lo) * (255.0 / (hi - lo + eps))
        return np.clip(y, 0, 255).astype(np.uint8)

    # signature matches your SDK: (void*, int64, int64, void*)
    def _onDataCallback(self, pBuffer: C.c_void_p,
                        nFrameSize: C.c_int64,
                        nFrameNumber: C.c_int64,
                        pContext: C.c_void_p) -> None:
        try:
            nbytes = int(nFrameSize)
            if nbytes <= 0:
                return
            line16, bands = self._reshape_line_from_bytes(pBuffer, nbytes)
            if line16 is None:
                return
            self._bands_in_line = bands
            if BAND_IDXS.max() >= bands:
                return

            rgb16 = line16[:, BAND_IDXS]  # (1024, 3) uint16
            # print (rgb16)
            # enqueue a COPY so we're safe after the SDK returns
            try:
                self.line_queue.put_nowait(rgb16.copy())
            except Exception:
                try:
                    self.line_queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.line_queue.put_nowait(rgb16.copy())
                except Exception:
                    pass

            self._lines_rcvd += 1
        except Exception:
            # never raise out of native callback thread
            return

    def _drain_and_update_plot(self):
        drained = 0
        # write pointer for rolling buffer
        if not hasattr(self, "_write_row"):
            self._write_row = 0

        while True:
            try:
                rgb16 = self.line_queue.get_nowait()
            except Empty:
                break
            rgb8 = self._u16_to_u8(rgb16)  # (1024, 3) uint8
            self.rgb_img[self._write_row, :, :] = rgb8
            self._write_row = (self._write_row + 1) % ROLLING_HEIGHT
            drained += 1

        if drained:
            # rotate so newest line is at the bottom (scroll effect)
            wr = self._write_row
            view = np.vstack((self.rgb_img[wr:], self.rgb_img[:wr]))
            self.im.set_data(view)
            # optional: show stats in window title or a label
            self.ax.set_title(
                f"HSI RGB bands {BAND_IDXS.tolist()} | "
                f"lines: {self._lines_rcvd}  bands_in_line: {self._bands_in_line}  "
                f"q: {self.line_queue.qsize()}"
            )
            self.canvas.draw_idle()

    def btnCameraConnect_clicked(self):
        self.btnCameraConnect.setEnabled(False)
        self.btnCameraDisconnect.setEnabled(True)

        QApplication.setOverrideCursor(Qt.WaitCursor) # provide user feedback that we are waiting for a process to finish via cursor wait icon
        self._start_sensor_and_callback()
        QApplication.restoreOverrideCursor()

    def btnCameraDisconnect_clicked(self):
        self.btnCameraDisconnect.setEnabled(False)
        self.btnCameraConnect.setEnabled(True)

        # track camera connection
        self.cam_connected = False

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
        print("Status: " + str(message))

    def btnStartAcquire_clicked(self):
        # check if we are connect to camera first
        if self.cam_connected == False:
            print("Status: Camera not connected")
            return
        
        # check rig controller is connected
        if self.rig_controller == None or not self.rig_controller.is_connected():
            print("Status: Bed controller not connected")
            # print(f"rig controller: {self.rig_controller}")
            # print(f"is connected? : {self.rig_controller.is_connected()}")
            return
        
        self.btnStartAcquire.setEnabled(False)
        self.btnStopAcquire.setEnabled(True)
        
        # Create worker thread
        self.scan_worker = ScanWorkerThread(
            main_window=self,
            rig_controller=self.rig_controller,
            cam_height=rig_settings.RIG_CAM_HEIGHT,
            init_pos=rig_settings.RIG_BED_START,
            end_scan_pos=rig_settings.RIG_BED_END,
            scan_speed=rig_settings.RIG_SPEED,
        )
        
        # Connect signals
        self.scan_worker.status_update.connect(self.on_scan_status_update)
        self.scan_worker.scan_complete.connect(self.on_scan_complete)
        self.scan_worker.error_occurred.connect(self.on_scan_error)
        
        # NEW: Connect confirmation request signal
        self.scan_worker.request_confirmation.connect(self.show_confirmation_dialog)
        
        # Start the thread
        self.scan_worker.start()
    @pyqtSlot(str)
    def show_confirmation_dialog(self, message):
        """
        Show a modal confirmation dialog in the main thread.
        This is called via signal from the worker thread.
        """
        reply = QMessageBox.question(
            self,
            'Confirmation Required',
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # User clicked Yes - resume the worker
            self.scan_worker.resume()
        else:
            # User clicked No - cancel the worker
            self.scan_worker.cancel()

    def on_scan_status_update(self, message):
        """Update status label with scan progress"""
        print(f"Status: {message}")
        
    def on_scan_complete(self):
        """Handle scan completion"""
        self.btnStartAcquire.setEnabled(True)
        self.btnStopAcquire.setEnabled(False)
        self.specSensor.command('Acquisition.Stop')
        print("Status: Scan completed")
        
        # Disconnect controller if it exists (NOTE: This should be user specified now, but there might be edge cases that might want this back)
        # if hasattr(self, 'scan_worker') and self.scan_worker.rig_controller:
        #     self.scan_worker.rig_controller.disconnect()
            
    def on_scan_error(self, error_message):
        """Handle scan errors"""
        self.btnStartAcquire.setEnabled(True)
        self.btnStopAcquire.setEnabled(False)
        print(f"Error: {error_message}")
        
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
        print("Status: Stopped")

    def closeEvent(self, event):
        # stop UI timer
        try:
            self.timer.stop()
        except Exception:
            pass
        # stop acquisition & close device
        try:
            self.specSensor.command('Acquisition.Stop')
        except Exception:
            pass
        try:
            self.specSensor.close()
        except Exception:
            pass
        super().closeEvent(event)


class ScanWorkerThread(QThread):
    # signal that enable comms back to gui main thread
    status_update = pyqtSignal(str)
    scan_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    # Signal to request user confirmation
    request_confirmation = pyqtSignal(str)  # str is the message to display

    def __init__(self, 
                 main_window: MainWindow, 
                 rig_controller: RIGController = None, # if None, then use the one in main_window 
                 cam_height: float=0.0, 
                 scan_speed: float=443.33, 
                 init_pos: float=80.0, 
                 end_scan_pos: float=650.0
                 ):
        super().__init__()
        self.main_window = main_window
        if rig_controller == None:
            self.rig_controller = main_window.rig_controller
        else:
            self.rig_controller = rig_controller
        self.cam_height = cam_height
        self.scan_speed = scan_speed
        self.init_pos = init_pos
        self.scan_pos = end_scan_pos
        self._is_running = True

        # Pause control of the routine for getting user input when calibration is enabled
        self._paused = False
        self._pause_mutex = QMutex()
        self._pause_condition = QWaitCondition()
        self._user_cancelled = False

    
    def stop(self):
        """Stop the scan routine"""
        self._is_running = False

        # if needed, wake the thread in case it’s waiting
        with QMutexLocker(self._pause_mutex):
            self._paused = False
            self._pause_condition.wakeAll()

        # check if we still have a controller that is connected first
        if self.rig_controller is not None and self.rig_controller.is_connected():
            self.rig_controller.emergency_stop() # software e-stop the controller
            self.rig_controller.disconnect() # disconnect from the controller
            
            self.main_window.btnRigConnect.setEnabled(True)
            self.main_window.btnRigDisconnect.setEnabled(False)
            self.main_window.btnRigConnect_2.setEnabled(True)
            self.main_window.btnRigDisconnect_2.setEnabled(False)

    @pyqtSlot()
    def resume(self):
        """Resume after user confirmation (Yes clicked)"""
        with QMutexLocker(self._pause_mutex):
            self._paused = False
            self._user_cancelled = False
            self._pause_condition.wakeAll()
    
    @pyqtSlot()
    def cancel(self):
        """Cancel after user rejection (No clicked)"""
        with QMutexLocker(self._pause_mutex):
            self._paused = False
            self._user_cancelled = True
            self._is_running = False
            self._pause_condition.wakeAll()

    def wait_for_confirmation(self, message):
            """
            Pause the thread and wait for user confirmation via GUI.
            Returns True if user confirmed, False if cancelled.
            """
            with QMutexLocker(self._pause_mutex):
                self._paused = True
                self._user_cancelled = False
                
            # Emit signal to main thread to show dialog
            self.request_confirmation.emit(message)
            
            # Wait for user response
            with QMutexLocker(self._pause_mutex):
                while self._paused and self._is_running:
                    self._pause_condition.wait(self._pause_mutex)
            
            return not self._user_cancelled and self._is_running
            
    def run(self):
        """This runs the scan routine on in a separate thread"""
        try:
            # rig_settings.RIG_TIMEOUT_READ_ONLY = 95
            # rig_settings.RIG_TRAVEL_SPEED_READ_ONLY = 6000
            
            self.status_update.emit("Starting scan routine...")
            
            if not self.rig_controller.serial_conn or not self.rig_controller.serial_conn.is_open:
                self.error_occurred.emit("ERROR: Not connected to controller!")
                return
                
            # Step 1: Reset and home
            self.status_update.emit("Resetting controller and homing axes...")
            self.rig_controller.reset_controller()
            time.sleep(2)
            
            response = self.rig_controller.home_axes()
            print(f"Homing responce during routine:\n {response}")
            
            # Wait for homing with interruptible sleep
            self.status_update.emit("Waiting for homing to complete...")
            start_time = time.time()
            while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY and self._is_running:
                pos = self.rig_controller.get_current_position()
                if pos is not None and pos["Y"] == 0.0 and pos["Z"] == 0.0:
                    break
                self.msleep(100)  # Use QThread's msleep for better integration
                
            if not self._is_running:
                return
            
            # Check to see if we need to move to calibration positions for black and white
            if self.main_window.chkCalibration.isChecked():
                print(f"Moving to white calibration strip")
                # NOTE: You may need to update the Y/Z positions for your strip in rig_settings.py if incorrect
                y_pos = rig_settings.RIG_WHITE_CAL_POS_READ_ONLY # Position still needs calibrating
                self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)  # Set feedrate in mm/min

                # Move to Y axis position (strip location)
                success_y = self.rig_controller.move_axis('Y', y_pos)
                
                # Wait for moving with interruptible sleep
                print("Waiting for move to white calibration strip to complete...")
                start_time = time.time()
                while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY:
                    pos = self.rig_controller.get_current_position()
                    if pos is not None and pos["Y"] == rig_settings.RIG_WHITE_CAL_POS_READ_ONLY and pos["Z"] == rig_settings.RIG_CAM_HEIGHT:
                        break
                    self.msleep(100)  # Use QThread's msleep for better integration
                    
                if success_y:
                    print("Status: At white calibration strip")
                else:
                    print("Status: Move failed")

                # Pause: dialog box here to get user confirmation to continue routine
                self.status_update.emit("At white calibration position. Waiting for user confirmation...")
                
                if not self.wait_for_confirmation("Calibration paused at white strip. Do you want to continue?"):
                    self.status_update.emit("Scan cancelled by user.")
                    return
                
                self.status_update.emit("Continuing scan routine...")

                print(f"Moving to black calibration strip")
                if self.rig_controller != None and not self.rig_controller.is_connected():
                    print("Status: Bed controller not connected")
                    return

                # NOTE: You may need to update the Y/Z positions for your strip in rig_settings.py if incorrect
                y_pos = rig_settings.RIG_BLACK_CAL_POS_READ_ONLY # Position still needs calibrating
                self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)  # Set feedrate in mm/min

                # Move to Y axis position (strip location)
                success_y = self.rig_controller.move_axis('Y', y_pos)
                
                # TODO: Move this homing to seperate thread process so that it is not blocking
                # Wait for moving with interruptible sleep
                print("Waiting for move to white calibration strip to complete...")
                start_time = time.time()
                while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY:
                    pos = self.rig_controller.get_current_position()
                    if pos is not None and pos["Y"] == rig_settings.RIG_BLACK_CAL_POS_READ_ONLY and pos["Z"] == rig_settings.RIG_CAM_HEIGHT:
                        break
                    self.msleep(100)  # Use QThread's msleep for better integration
                    
                if success_y:
                    print("Status: At white calibration strip")
                else:
                    print("Status: Move failed")
                # Pause: dialog box here to get user confirmation to continue routine
                self.status_update.emit("At black calibration position. Waiting for user confirmation...")
                
                if not self.wait_for_confirmation("Calibration paused at black strip. Do you want to continue?"):
                    self.status_update.emit("Scan cancelled by user.")
                    return
                
                self.status_update.emit("Continuing scan routine...")
                
                if not self._is_running:
                    return
                
            # Move to initial position
            self.status_update.emit("Moving to initial position...")
            self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)
            self.rig_controller.move_axis("Y", self.init_pos)
            
            start_time = time.time()
            while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY and self._is_running:
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
            while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY and self._is_running:
                pos = self.rig_controller.get_current_position()
                if pos is not None and pos["Y"] == self.scan_pos and pos["Z"] == self.cam_height:
                    break
                self.msleep(100)
                
            if not self._is_running:
                return
                
            # Return to initial position
            self.status_update.emit("Returning to initial position...")
            self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)
            self.rig_controller.move_axis("Y", self.init_pos)
            
            start_time = time.time()
            while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY and self._is_running:
                pos = self.rig_controller.get_current_position()
                if pos is not None and pos["Y"] == self.init_pos and pos["Z"] == self.cam_height:
                    break
                self.msleep(100)
                
            self.status_update.emit("Scan routine completed successfully!")
            self.scan_complete.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Error during scan: {str(e)}")

