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
import math
from controllers.command_client import send_command
from controllers.rig_controller import RIGController

import rig_settings

from utility import load_settings, save_settings
import glob
import gc
import sys
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
from scipy.io import savemat
from envi_io import write_envi_bil, load_envi_cube_if_exists

WIDTH = 1024  # Spatial width of the camera line
CAMERA_LENSE_FOV = 38.0
BAND_IDXS = np.array([202, 120, 0], dtype=int)  # R,G,B (0-based)
ROLLING_HEIGHT = 512
QUEUE_MAX = 4096
READOUT_TIME= 0.09 #milliseconds (this overwritten by query camera)

# 1 line is Approx. this pitch distance mm distance
LINE_PITCH = 0 # this is calculated later via the equation: (2 * height * math.tan(CAMERA_LENSE_FOV / 2)) / WIDTH

buffer_list = list()
white_buffer = list()
dark_buffer = list()

def find_rgb_bands(lam: np.ndarray) -> np.ndarray:
    """
    Port of MATLAB findRGBbands(lambda).
    Returns 0-based indices [R, G, B] based on wavelength thresholds.

    MATLAB expects lambda in micrometers (~0.4..0.7). If lam looks like nm (e.g. 450..900),
    we convert to micrometers by /1000.
    """
    lam = np.asarray(lam, dtype=float).ravel()

    # Heuristic: if wavelengths look like nm, convert to um
    if lam.size and lam.max() > 5.0:
        lam = lam / 1000.0

    red   = 0.6329
    green = 0.5510
    blue  = 0.454528

    # MATLAB: B=max(find(lambda<=blue)); if empty B=1
    b_idx = np.where(lam <= blue)[0]
    B = int(b_idx.max()) if b_idx.size else 0  # MATLAB 1 -> python 0

    # MATLAB: R=max(find(lambda<=red)); if empty R=end
    r_idx = np.where(lam <= red)[0]
    R = int(r_idx.max()) if r_idx.size else int(lam.size - 1)

    # MATLAB: G=max(find(lambda<=green)); if empty G=2
    g_idx = np.where(lam <= green)[0]
    G = int(g_idx.max()) if g_idx.size else 1  # MATLAB 2 -> python 1

    return np.array([R, G, B], dtype=int)


def make_rgb_image(image_cube: np.ndarray,
                   lam: np.ndarray,
                   brightness: float = 1.0,
                   eps: float = 1e-12,
                   clip01: bool = True) -> np.ndarray:
    """
    UZ:Port of my MATLAB makeRGBimage(imageCube, lambda, brightness).

    image_cube: H x W x B
    lam: vector length B
    brightness: MATLAB default 1.25
    Returns float RGB in [0,1] if clip01=True.
    """
    cube = np.asarray(image_cube)
    if cube.ndim != 3:
        raise ValueError("image_cube must be 3D: H x W x Bands")

    _, _, B = cube.shape
    out = find_rgb_bands(lam)
    if out.max() >= B:
        raise ValueError(f"RGB band index out of range. bands={B}, out={out.tolist()}")

    rgb = cube[:, :, out].astype(np.float32)  # H x W x 3

    # MATLAB normalization per channel:
    #   ch = ch - min(ch)
    #   ch = (brightness * ch) / max(ch)
    for i in range(3):
        ch = rgb[:, :, i]
        ch = ch - np.min(ch)
        mx = np.max(ch)
        if mx > eps:
            ch = (brightness * ch) / mx
        else:
            ch = ch * 0.0
        if clip01:
            ch = np.clip(ch, 0.0, 1.0)
        rgb[:, :, i] = ch

    return rgb


def make_rgb_line(line_pixels_by_band: np.ndarray,
                  lam: np.ndarray,
                  brightness: float = 1.25) -> np.ndarray:
    """
    Helper for linescan:
    line_pixels_by_band: (WIDTH, bands) uint16/float/...
    Returns uint8 RGB: (WIDTH, 3)
    """
    line = np.asarray(line_pixels_by_band)
    if line.ndim != 2:
        raise ValueError("line must be 2D: (pixels, bands)")

    cube = line[None, :, :]                       # 1 x WIDTH x bands
    rgb01 = make_rgb_image(cube, lam, brightness) # 1 x WIDTH x 3 float [0..1]
    return (rgb01[0] * 255.0).astype(np.uint8)    # WIDTH x 3 uint8




# class
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Track what tab we are on
        self.current_tab = 0
        # track if camera is connected (NOTE: Might be better way to get this from SDK request)
        self.cam_connected = False



        # load Settings from YAML
        self.loading_settings()

        self.add_camera_preview_ui()

        # rig controller object that should be used when connecting and commanding the rig
        self.rig_controller: RIGController = None

        # Set up the `setup` tab in the UI for the Rig Control and connections
        self.setup_rig_ui()

        # setup the buttons for the UI
        self.configure_buttons()

        self.white_calibration_data = False
        self.dark_calibration_data = False

        self.apply_calibration=True #False

        self.acquisition_stopped=False
        self.line_count=0



    def _current_output_dir(self) -> str:
        """Single source of truth for where to load/save calibration."""
        try:
            d = self.txtOutputFolderPath.text().strip()
            if d:
                return d
        except Exception:
            pass
        return str(getattr(rig_settings, "OUTPUT_FOLDER", "."))

    def _find_latest_envi_base(self, folder: str, prefix: str) -> str | None:
        """
        Find latest ENVI base path for files named like:
          {prefix}{timestamp}.hdr + {prefix}{timestamp}.bil
        Returns base path without extension, or None.
        """
        pattern = os.path.join(folder, f"{prefix}*.hdr")
        hdrs = glob.glob(pattern)
        if not hdrs:
            return None

        # sort by modified time (most robust even if timestamps collide)
        hdrs.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        for hdr in hdrs:
            base = os.path.splitext(hdr)[0]  # strip .hdr
            bil = base + ".bil"
            if os.path.exists(bil):
                return base
        return None

    def load_latest_calibration(self) -> None:
        """
        Load *latest* white/dark calibration from the output folder.
        Updates self.white_ref / self.dark_ref.
        Safe to call many times.
        """
        out_dir = self._current_output_dir()
        os.makedirs(out_dir, exist_ok=True)

        try:
            # Prefer finding latest ourselves, then use your loader.
            # If load_envi_cube_if_exists already finds latest, this still forces correct folder.
            w_base = self._find_latest_envi_base(out_dir, "white_calib_")
            d_base = self._find_latest_envi_base(out_dir, "dark_calib_")

            # Fallback to your existing helper if needed
            w_cube = d_cube = None
            w_hdr = d_hdr = None

            if w_base is not None:
                # If your load_envi_cube_if_exists only supports (dir, prefix), keep this:
                w_cube, _, w_hdr = load_envi_cube_if_exists(out_dir, "white_calib_")
            else:
                w_cube, _, w_hdr = load_envi_cube_if_exists(out_dir, "white_calib_")

            if d_base is not None:
                d_cube, _, d_hdr = load_envi_cube_if_exists(out_dir, "dark_calib_")
            else:
                d_cube, _, d_hdr = load_envi_cube_if_exists(out_dir, "dark_calib_")

            if w_cube is not None:
                # (lines, width, bands) -> mean over lines -> (width,bands) OR mean over both -> (bands,)
                # Your ELM expects (width,bands), so use mean(axis=0)
                self.white_ref = w_cube.mean(axis=0).astype(np.float32)
                print(f"Loaded latest white calibration from: {w_hdr}")

            if d_cube is not None:
                self.dark_ref = d_cube.mean(axis=0).astype(np.float32)
                print(f"Loaded latest dark calibration from: {d_hdr}")

        except Exception as e:
            print(f"Calibration load failed: {e}")

    def _drain_queue(self, q):
        """Remove all pending items so they don't keep memory alive."""
        try:
            while True:
                q.get_nowait()
        except Exception:
            pass

    def _clear_memory_after_save(self, aggressive: bool = True):
        """
        Explicitly release references to large buffers and force GC.
        Call this right after you finish saving a scan/calibration.
        """
        # Clear big line buffers
        try:
            self.hsi_lines.clear()
        except Exception:
            self.hsi_lines = []

        try:
            self.white_cal_lines.clear()
        except Exception:
            self.white_cal_lines = []

        try:
            self.dark_cal_lines.clear()
        except Exception:
            self.dark_cal_lines = []

        # Drain queue (important: queue holds numpy arrays)
        try:
            if hasattr(self, "line_queue") and self.line_queue is not None:
                self._drain_queue(self.line_queue)
        except Exception:
            pass

        # Drop rolling preview view if you want to aggressively free
        # (optional; normally rgb_img is small-ish, but keep it if needed)
        if aggressive:
            try:
                # Keep the preview array allocated (fast) OR uncomment to reinit
                # self.rgb_img = np.zeros((ROLLING_HEIGHT, WIDTH, 3), dtype=np.uint8)
                pass
            except Exception:
                pass

        # Force GC
        gc.collect()

        # OPTIONAL: On Linux, return freed heap pages to OS
        # (No-op on Windows; safe to wrap)
        if sys.platform.startswith("linux"):
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass

    def _apply_elm(self, line16: np.ndarray, eps_dn: float = 1.0) -> np.ndarray:
            """
            calibrated = (raw - dark) / (white - dark + eps)
            If refs missing -> return raw uint16 (so preview uses raw rendering).
            Returns float32 reflectance in [0,1] when calibrated.
            """
            if self.white_ref is None or self.dark_ref is None:
                return line16  # IMPORTANT: keep as uint16

            raw = line16.astype(np.float32)
            white = self.white_ref.astype(np.float32)
            dark = self.dark_ref.astype(np.float32)

            denom = white - dark
            denom = np.maximum(denom, eps_dn)  # DN-scale epsilon

            corr = (raw - dark) / denom
            corr = np.clip(corr, 0.0, 1.0).astype(np.float32)
            return corr

    def _finalize_and_save_calibration(self, kind: str):
        """
        kind: "white" or "dark"
        Builds calibration reference from collected lines, saves ENVI, and updates refs.
        """
        try:
            rig_settings.OUTPUT_FOLDER = self.txtOutputFolderPath.text()
            out_dir = getattr(rig_settings, "OUTPUT_FOLDER", ".")
            os.makedirs(out_dir, exist_ok=True)

            if kind == "white":
                if len(self.white_cal_lines) == 0:
                    return
                cube = np.stack(self.white_cal_lines, axis=0)  # (lines, WIDTH, bands)
                # save cube as ENVI BIL
                ts = time.strftime("%Y%m%d_%H%M%S")
                base_path = os.path.join(out_dir, f"white_calib_{ts}")
                write_envi_bil(
                    base_path=base_path,
                    cube=cube.astype(np.uint16, copy=False),
                    wavelength=self.lambda_vec if (
                                self.lambda_vec is not None and len(self.lambda_vec) == cube.shape[2]) else None,
                    description="White calibration (BIL)",
                )
                # update reference (mean over lines and samples)
                #self.white_ref = cube.mean(axis=(0, 1)).astype(np.float32)
                self.white_ref = cube.mean(axis=0).astype(np.float32)
                #self.dark_ref = cube.mean(axis=0).astype(np.float32)
                print(f"Saved white calibration ENVI: {base_path}.bil/.hdr")
                self.white_cal_lines = []  # clear after saving

            elif kind == "dark":
                if len(self.dark_cal_lines) == 0:
                    return
                cube = np.stack(self.dark_cal_lines, axis=0)
                ts = time.strftime("%Y%m%d_%H%M%S")
                base_path = os.path.join(out_dir, f"dark_calib_{ts}")
                write_envi_bil(
                    base_path=base_path,
                    cube=cube.astype(np.uint16, copy=False),
                    wavelength=self.lambda_vec if (
                                self.lambda_vec is not None and len(self.lambda_vec) == cube.shape[2]) else None,
                    description="Dark calibration (BIL)",
                )
                self.dark_ref = cube.mean(axis=(0, 1)).astype(np.float32)
                print(f"Saved dark calibration ENVI: {base_path}.bil/.hdr")
                self.dark_cal_lines = []

        except Exception as e:
            print(f"Calibration save failed ({kind}): {e}")

    def add_camera_preview_ui(self):
        # --- replace QLabel with Matplotlib canvas ---
        self.hsi_lines = []  # list of (WIDTH, bands) lines collected during scan
        # --- calibration buffers (collected during calibration pauses) ---
        self.white_cal_lines = []   # list of (WIDTH, bands)
        self.dark_cal_lines = []    # list of (WIDTH, bands)

        # references used for ELM (each is (bands,) float32)
        self.white_ref = None
        self.dark_ref = None

        # internal flag to know if we should finalize/save when modes change
        self._prev_cal_mode = None  # None / "white" / "dark"


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

        # ---- Added for MATLAB-style RGB from wavelength vector ----
        # Wavelength vector (length must equal bands in each line). Set this after opening camera.
        self.lambda_vec = np.asarray([
    400.00,401.34,402.68,404.02,405.36,406.70,408.04,409.38,410.71,412.05,413.39,414.73,
    416.07,417.41,418.75,420.09,421.43,422.77,424.11,425.45,426.79,428.12,429.46,430.80,
    432.14,433.48,434.82,436.16,437.50,438.84,440.18,441.52,442.86,444.20,445.54,446.88,
    448.21,449.55,450.89,452.23,453.57,454.91,456.25,457.59,458.93,460.27,461.61,462.95,
    464.29,465.62,466.96,468.30,469.64,470.98,472.32,473.66,475.00,476.34,477.68,479.02,
    480.36,481.70,483.04,484.38,485.71,487.05,488.39,489.73,491.07,492.41,493.75,495.09,
    496.43,497.77,499.11,500.45,501.79,503.12,504.46,505.80,507.14,508.48,509.82,511.16,
    512.50,513.84,515.18,516.52,517.86,519.20,520.54,521.88,523.21,524.55,525.89,527.23,
    528.57,529.91,531.25,532.59,533.93,535.27,536.61,537.95,539.29,540.62,541.96,543.30,
    544.64,545.98,547.32,548.66,550.00,551.34,552.68,554.02,555.36,556.70,558.04,559.38,
    560.71,562.05,563.39,564.73,566.07,567.41,568.75,570.09,571.43,572.77,574.11,575.45,
    576.79,578.12,579.46,580.80,582.14,583.48,584.82,586.16,587.50,588.84,590.18,591.52,
    592.86,594.20,595.54,596.88,598.21,599.55,600.89,602.23,603.57,604.91,606.25,607.59,
    608.93,610.27,611.61,612.95,614.29,615.62,616.96,618.30,619.64,620.98,622.32,623.66,
    625.00,626.34,627.68,629.02,630.36,631.70,633.04,634.38,635.71,637.05,638.39,639.73,
    641.07,642.41,643.75,645.09,646.43,647.77,649.11,650.45,651.79,653.12,654.46,655.80,
    657.14,658.48,659.82,661.16,662.50,663.84,665.18,666.52,667.86,669.20,670.54,671.88,
    673.21,674.55,675.89,677.23,678.57,679.91,681.25,682.59,683.93,685.27,686.61,687.95,
    689.29,690.62,691.96,693.30,694.64,695.98,697.32,698.66,700.00,701.34,702.68,704.02,
    705.36,706.70,708.04,709.38,710.71,712.05,713.39,714.73,716.07,717.41,718.75,720.09,
    721.43,722.77,724.11,725.45,726.79,728.12,729.46,730.80,732.14,733.48,734.82,736.16,
    737.50,738.84,740.18,741.52,742.86,744.20,745.54,746.88,748.21,749.55,750.89,752.23,
    753.57,754.91,756.25,757.59,758.93,760.27,761.61,762.95,764.29,765.62,766.96,768.30,
    769.64,770.98,772.32,773.66,775.00,776.34,777.68,779.02,780.36,781.70,783.04,784.38,
    785.71,787.05,788.39,789.73,791.07,792.41,793.75,795.09,796.43,797.77,799.11,800.45,
    801.79,803.12,804.46,805.80,807.14,808.48,809.82,811.16,812.50,813.84,815.18,816.52,
    817.86,819.20,820.54,821.88,823.21,824.55,825.89,827.23,828.57,829.91,831.25,832.59,
    833.93,835.27,836.61,837.95,839.29,840.62,841.96,843.30,844.64,845.98,847.32,848.66,
    850.00,851.34,852.68,854.02,855.36,856.70,858.04,859.38,860.71,862.05,863.39,864.73,
    866.07,867.41,868.75,870.09,871.43,872.77,874.11,875.45,876.79,878.12,879.46,880.80,
    882.14,883.48,884.82,886.16,887.50,888.84,890.18,891.52,892.86,894.20,895.54,896.88,
    898.21,899.55,900.89,902.23,903.57,904.91,906.25,907.59,908.93,910.27,911.61,912.95,
    914.29,915.62,916.96,918.30,919.64,920.98,922.32,923.66,925.00,926.34,927.68,929.02,
    930.36,931.70,933.04,934.38,935.71,937.05,938.39,939.73,941.07,942.41,943.75,945.09,
    946.43,947.77,949.11,950.45,951.79,953.12,954.46,955.80,957.14,958.48,959.82,961.16,
    962.50,963.84,965.18,966.52,967.86,969.20,970.54,971.88,973.21,974.55,975.89,977.23,
    978.57,979.91,981.25,982.59,983.93,985.27,986.61,987.95,989.29,990.62,991.96,993.30,
    994.64,995.98,997.32,998.66], dtype=np.float32)


        # Cache for indices returned by find_rgb_bands(lambda_vec)
        self._rgb_band_idxs = None

        # MATLAB default brightness (can be tuned)
        self.rgb_brightness = 1.0


        # load previous calibration from disk (if exists)
        try:
            out_dir = getattr(rig_settings, "OUTPUT_FOLDER", ".")
            os.makedirs(out_dir, exist_ok=True)

            w_cube, _, w_hdr = load_envi_cube_if_exists(out_dir, "white_calib_")
            d_cube, _, d_hdr = load_envi_cube_if_exists(out_dir, "dark_calib_")

            # Expect calibration saved as cube with lines >=1, samples=WIDTH, bands=B
            if w_cube is not None:
                # make white_ref as mean over lines and samples -> (bands,)
                self.white_ref = w_cube.mean(axis=(0, 1)).astype(np.float32)
                print(f"Loaded white calibration: {w_hdr}")

            if d_cube is not None:
                self.dark_ref = d_cube.mean(axis=(0, 1)).astype(np.float32)
                print(f"Loaded dark calibration: {d_hdr}")

        except Exception as e:
            print(f"Calibration load skipped: {e}")



        # start camera and register callback
        # self._start_sensor_and_callback()

        # UI timer to update plot (main thread only)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._drain_and_update_plot)
        self.timer.start(30)  # ~33 FPS UI
        # --- end of eplace QLabel with Matplotlib canvas ---

    def setup_rig_ui(self):

        # get the available com ports and add them to combobox list
        self.update_comm_port_list()
        # lets also udate the list everytime we change to the setup tab
        self.tabWidget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        # tab 1 (index=1) is the Setup tab for the Rig
        self.current_tab = index  # update the current tab index we are on (NOTE: there is probably a better pyQT way of getting this info)
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

        # update settings button link
        self.btnUpdateSettings.clicked.connect(self.update_settings)

        # Rig controller buttons configure
        self.btnHomeBed.clicked.connect(self.home_bed_clicked)
        self.btnHomeCarriage.clicked.connect(self.home_carriage_clicked)
        self.btnWhiteStrip.clicked.connect(self.move_to_white_calibration)
        self.btnBlackStrip.clicked.connect(self.move_to_black_calibration)
        # self.btnScan.clicked.connect(self.start_scan)
        # self.btnStop.clicked.connect(self.stop)
        self.btnReset.clicked.connect(self.reset_controller)

    # def calcSpeedFromFPS(self, FPS=50):
    #     speed = 0.0  # Speed in mm/min
    #     fps = FPS  # frame rate for the camera to capture at
    #     # This is `mm`
    #     height = rig_settings.RIG_CAM_HEIGHT_OFFSET + rig_settings.RIG_CAM_HEIGHT
    #     spatial_width = WIDTH

    #     # Multiply by 60 to get speed in mm/min instead of mm/sec
    #     fov_rad = math.radians(CAMERA_LENSE_FOV)

    #     speed = 60*fps * ((2 * height * math.tan(fov_rad / 2)) / spatial_width)
    #     # speed_offset = 16.5
    #     # speed = speed + speed_offset
    #     # speed = 60 * fps * LINE_PITCH
    #     print(f"calculated speed from FPS: {speed}")

    #     return speed
    def calcSpeedFromFPS(self, FPS=50):
        # This is `mm`
        height = rig_settings.RIG_CAM_HEIGHT_OFFSET + rig_settings.RIG_CAM_HEIGHT

        object_length = (0.6842 * height) + 21.368

        actual_pitch = object_length / WIDTH

        speed = 60 * FPS * actual_pitch
        print(f"calculated speed from FPS: {speed}")

        return speed

    def loading_settings(self):
        """
        Load settings from config/settings.yaml into rig_settings and populate UI fields.
        """
        try:
            settings = load_settings()  # loads into rig_settings as well
        except Exception as e:
            print(f"Warning: failed to load settings.yaml: {e}")
            settings = {}

        # Populate UI text fields if those widgets exist
        try:
            # Rig values
            if hasattr(self, "txtSpeed"):
                self.txtSpeed.setText(str(getattr(rig_settings, "RIG_SPEED", "")))
            if hasattr(self, "txtBedStartPosition"):
                self.txtBedStartPosition.setText(str(getattr(rig_settings, "RIG_BED_START", "")))
            if hasattr(self, "txtBedEndPosition"):
                self.txtBedEndPosition.setText(str(getattr(rig_settings, "RIG_BED_END", "")))
            if hasattr(self, "txtCameraPosition"):
                self.txtCameraPosition.setText(str(getattr(rig_settings, "RIG_CAM_HEIGHT", "")))

            # Camera / acquisition values
            if hasattr(self, "txtOutputFolderPath"):
                self.txtOutputFolderPath.setText(str(getattr(rig_settings, "OUTPUT_FOLDER", "")))

            if hasattr(self, "textEditFrameRate"):
                # QTextEdit or similar; prefer plain text setter if available
                try:
                    self.textEditFrameRate.setPlainText(str(getattr(rig_settings, "CAMERA_FRAME_RATE", "")))
                except Exception:
                    try:
                        self.textEditFrameRate.setText(str(getattr(rig_settings, "CAMERA_FRAME_RATE", "")))
                    except Exception:
                        pass

            if hasattr(self, "textEditExposure"):
                try:
                    self.textEditExposure.setPlainText(str(getattr(rig_settings, "CAMERA_EXPOSURE", "")))
                except Exception:
                    try:
                        self.textEditExposure.setText(str(getattr(rig_settings, "CAMERA_EXPOSURE", "")))
                    except Exception:
                        pass

            if hasattr(self, "textEditFrameCount"):
                try:
                    self.textEditFrameCount.setPlainText(str(getattr(rig_settings, "CAMERA_LINE_COUNT", "")))
                except Exception:
                    try:
                        self.textEditFrameCount.setText(str(getattr(rig_settings, "CAMERA_LINE_COUNT", "")))
                    except Exception:
                        pass

            # Combobox helpers: try to match item text to saved value, fallback to index 0
            def set_combobox_to_value(cmb, val, human_name=None):
                """
                Try to set combobox to item whose text equals val.
                If not found, set index 0 and show a warning dialog to the user
                (only when val is non-empty) to indicate the loaded value was invalid.
                Returns True if matched, False if fallback used.
                """
                try:
                    target = str(val)
                    for i in range(cmb.count()):
                        if str(cmb.itemText(i)) == target:
                            cmb.setCurrentIndex(i)
                            return True
                except Exception:
                    pass
                # not found -> fallback to index 0
                try:
                    cmb.setCurrentIndex(0)
                except Exception:
                    pass
                # show warning if a non-empty value failed to match
                try:
                    if val is not None and str(val) != "":
                        name = human_name or cmb.objectName()
                        QMessageBox.warning(
                            self,
                            "Settings load",
                            f"Could not load saved value '{val}' for '{name}'. Using default selection."
                        )
                except Exception:
                    pass
                return False

            if hasattr(self, "cmbShutter"):
                shutter = str(getattr(rig_settings, "CAMERA_SHUTTER", "Open"))
                # try match text, else set by simple open/close
                set_combobox_to_value(self.cmbShutter, shutter, "Camera shutter")
                try:
                    if self.cmbShutter.currentIndex() not in range(self.cmbShutter.count()):
                        self.cmbShutter.setCurrentIndex(0 if shutter.lower().startswith("open") else 1)
                except Exception:
                    pass

            if hasattr(self, "cmbSpectralBin"):
                set_combobox_to_value(self.cmbSpectralBin, getattr(rig_settings, "CAMERA_SPECTRAL_BIN", 1),
                                      "Spectral bin")

            if hasattr(self, "cmbSpatialBin"):
                set_combobox_to_value(self.cmbSpatialBin, getattr(rig_settings, "CAMERA_SPATIAL_BIN", 1), "Spatial bin")

            if hasattr(self, "cmbCaptureMode"):
                set_combobox_to_value(self.cmbCaptureMode, getattr(rig_settings, "CAMERA_CAPTURE_MODE", 1),
                                      "Capture mode")

        except Exception:
            pass

    def update_settings(self):
        # Saves update and save setting to YAML file

        self.update_camera_settings()
        self.update_rig_settings()

    # saving rig controller scanning config to globally accessable python config file and yaml file
    def update_rig_settings(self):

        # Update rig_settings values from UI and save to YAML
        def update_txt_value(attr_name, txtbox):
            value = None
            try:
                txt = txtbox.text()
                if str(txt) != "":
                    value = float(txt)
            except Exception:
                value = getattr(rig_settings, attr_name, None)
            if value is not None:
                setattr(rig_settings, attr_name, value)
            return value

        update_txt_value("RIG_SPEED", self.txtSpeed)
        update_txt_value("RIG_BED_START", self.txtBedStartPosition)
        update_txt_value("RIG_BED_END", self.txtBedEndPosition)
        update_txt_value("RIG_CAM_HEIGHT", self.txtCameraPosition)

        if self.current_tab == 0:
            # update rig speed based on FPS (Prioritising FPS caluculated Speed over user entered speed when on main tab)
            rig_settings.RIG_SPEED = self.calcSpeedFromFPS(rig_settings.CAMERA_FRAME_RATE)

        # Build dict and save
        settings_to_save = {
            "RIG_SPEED": getattr(rig_settings, "RIG_SPEED", None),
            "RIG_BED_START": getattr(rig_settings, "RIG_BED_START", None),
            "RIG_BED_END": getattr(rig_settings, "RIG_BED_END", None),
            "RIG_CAM_HEIGHT": getattr(rig_settings, "RIG_CAM_HEIGHT", None),
            "RIG_WHITE_CAL_POS_READ_ONLY": getattr(rig_settings, "RIG_WHITE_CAL_POS_READ_ONLY", None),
            "RIG_BLACK_CAL_POS_READ_ONLY": getattr(rig_settings, "RIG_BLACK_CAL_POS_READ_ONLY", None),
            "RIG_TRAVEL_SPEED_READ_ONLY": getattr(rig_settings, "RIG_TRAVEL_SPEED_READ_ONLY", None),
            "RIG_TIMEOUT_READ_ONLY": getattr(rig_settings, "RIG_TIMEOUT_READ_ONLY", None),
        }
        try:
            save_settings(settings_to_save)
        except Exception as e:
            print(f"Warning: failed to save settings.yaml: {e}")

        # debug prints
        print(f"rig speed: {rig_settings.RIG_SPEED}")
        print(f"rig bed start: {rig_settings.RIG_BED_START}")
        print(f"rig bed end: {rig_settings.RIG_BED_END}")
        print(f"rig cam height: {rig_settings.RIG_CAM_HEIGHT}")
        print(f"settings updated to device")

    def update_camera_settings(self):
        # Read UI camera controls and save to YAML via save_settings()
        try:
            # shutter
            cam_shutter = getattr(rig_settings, "CAMERA_SHUTTER", "Open")
            if hasattr(self, "cmbShutter"):
                try:
                    cam_shutter = self.cmbShutter.currentText()
                except Exception:
                    cam_shutter = self.cmbShutter.currentIndex()

            # frame rate
            frame_rate = getattr(rig_settings, "CAMERA_FRAME_RATE", None)
            if hasattr(self, "textEditFrameRate"):
                try:
                    frame_rate = float(self.textEditFrameRate.toPlainText().strip())
                except Exception:
                    try:
                        frame_rate = float(self.textEditFrameRate.text().strip())
                    except Exception:
                        pass

            # exposure
            exposure = getattr(rig_settings, "CAMERA_EXPOSURE", None)
            if hasattr(self, "textEditExposure"):
                try:
                    exposure = float(self.textEditExposure.toPlainText().strip())
                except Exception:
                    try:
                        exposure = float(self.textEditExposure.text().strip())
                    except Exception:
                        pass

            # spectral bin
            spectral_bin = getattr(rig_settings, "CAMERA_SPECTRAL_BIN", 1)
            if hasattr(self, "cmbSpectralBin"):
                try:
                    spectral_bin = int(self.cmbSpectralBin.currentText())
                except Exception:
                    try:
                        spectral_bin = int(self.cmbSpectralBin.currentIndex())
                    except Exception:
                        pass

            # spatial bin
            spatial_bin = getattr(rig_settings, "CAMERA_SPATIAL_BIN", 1)
            if hasattr(self, "cmbSpatialBin"):
                try:
                    spatial_bin = int(self.cmbSpatialBin.currentText())
                except Exception:
                    try:
                        spatial_bin = int(self.cmbSpatialBin.currentIndex())
                    except Exception:
                        pass

            # capture mode
            capture_mode = getattr(rig_settings, "CAMERA_CAPTURE_MODE", 1)
            if hasattr(self, "cmbCaptureMode"):
                try:
                    capture_mode = int(self.cmbCaptureMode.currentText())
                except Exception:
                    try:
                        capture_mode = int(self.cmbCaptureMode.currentIndex())
                    except Exception:
                        pass

            # line/frame count
            self.line_count = getattr(rig_settings, "CAMERA_LINE_COUNT", 0)
            if hasattr(self, "textEditFrameCount"):
                try:
                    self.line_count = int(self.textEditFrameCount.toPlainText().strip())
                except Exception:
                    try:
                        self.line_count = int(self.textEditFrameCount.text().strip())
                    except Exception:
                        pass

            # output folder
            output_folder = getattr(rig_settings, "OUTPUT_FOLDER", "")
            if hasattr(self, "txtOutputFolderPath"):
                try:
                    output_folder = self.txtOutputFolderPath.text().strip()
                except Exception:
                    pass

            # Apply to rig_settings
            setattr(rig_settings, "CAMERA_SHUTTER", cam_shutter)
            setattr(rig_settings, "CAMERA_FRAME_RATE", frame_rate)
            setattr(rig_settings, "CAMERA_EXPOSURE", exposure)
            setattr(rig_settings, "CAMERA_SPECTRAL_BIN", spectral_bin)
            setattr(rig_settings, "CAMERA_SPATIAL_BIN", spatial_bin)
            setattr(rig_settings, "CAMERA_CAPTURE_MODE", capture_mode)
            setattr(rig_settings, "CAMERA_LINE_COUNT", self.line_count)
            setattr(rig_settings, "OUTPUT_FOLDER", output_folder)

            # Build dict to save (include rig keys so file is merged cleanly)
            settings_to_save = {
                "CAMERA_SHUTTER": cam_shutter,
                "CAMERA_FRAME_RATE": frame_rate,
                "CAMERA_EXPOSURE": exposure,
                "CAMERA_SPECTRAL_BIN": spectral_bin,
                "CAMERA_SPATIAL_BIN": spatial_bin,
                "CAMERA_CAPTURE_MODE": capture_mode,
                "CAMERA_LINE_COUNT": self.line_count,
                "OUTPUT_FOLDER": output_folder,
                "RIG_SPEED": getattr(rig_settings, "RIG_SPEED", None),
                "RIG_BED_START": getattr(rig_settings, "RIG_BED_START", None),
                "RIG_BED_END": getattr(rig_settings, "RIG_BED_END", None),
                "RIG_CAM_HEIGHT": getattr(rig_settings, "RIG_CAM_HEIGHT", None),
                "RIG_WHITE_CAL_POS_READ_ONLY": getattr(rig_settings, "RIG_WHITE_CAL_POS_READ_ONLY", None),
                "RIG_BLACK_CAL_POS_READ_ONLY": getattr(rig_settings, "RIG_BLACK_CAL_POS_READ_ONLY", None),
                "RIG_TRAVEL_SPEED_READ_ONLY": getattr(rig_settings, "RIG_TRAVEL_SPEED_READ_ONLY", None),
                "RIG_TIMEOUT_READ_ONLY": getattr(rig_settings, "RIG_TIMEOUT_READ_ONLY", None),
            }

            save_settings(settings_to_save)
            print(
                f"Saved camera settings: shutter={cam_shutter}, fr={frame_rate}, exp={exposure}, sb={spectral_bin}, spb={spatial_bin}, mode={capture_mode}, lines={self.line_count}, out={output_folder}")
        except Exception as e:
            print(f"Warning: failed to save camera settings: {e}")

    # ===================
    # === RIG CODE ===
    # ===================

    # connecting to selected COM port from dropdown list
    def connect_controller(self):
        if self.current_tab == 0:  # if we are on the camera connection tab
            selected_index = self.cmbBoxCommPortSelect_2.currentIndex()
            port_info = self.cmbBoxCommPortSelect_2.itemData(selected_index)
        else:  # else we are on the rig control tab
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
            self.rig_controller = None  # clear reference to object after disconnect
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
        y_pos = rig_settings.RIG_WHITE_CAL_POS_READ_ONLY  # Position still needs calibrating
        self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)  # Set feedrate in mm/min

        # Move to Y axis position (strip location)
        success_y = self.rig_controller.move_axis('Y', y_pos)

        # TODO: Move this homing to seperate thread process so that it is not blocking
        # Wait for moving with interruptible sleep
        print("Waiting for move to white calibration strip to complete...")
        start_time = time.time()
        while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY:
            pos = self.rig_controller.get_current_position()
            if pos is not None and pos["Y"] == rig_settings.RIG_WHITE_CAL_POS_READ_ONLY and pos[
                "Z"] == rig_settings.RIG_CAM_HEIGHT:
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
        y_pos = rig_settings.RIG_BLACK_CAL_POS_READ_ONLY  # Position still needs calibrating
        z_pos = 0.0
        self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)  # Set feedrate in mm/min

        # Move to Y axis position (strip location)
        success_y = self.rig_controller.move_axis('Y', y_pos)
        success_z = self.rig_controller.move_axis('Z', z_pos)

        # TODO: Move this homing to seperate thread process so that it is not blocking
        # Wait for moving with interruptible sleep
        print("Waiting for move to white calibration strip to complete...")
        start_time = time.time()
        while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY:
            pos = self.rig_controller.get_current_position()
            if pos is not None and pos["Y"] == rig_settings.RIG_BLACK_CAL_POS_READ_ONLY and pos["Z"] == z_pos:
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

        # update settings to camera
        try:
            self.updateSettingsToDevice()
        except Exception as e:
            print(f"Error: failed to update settings to device: {e}")

        # start acquisition
        # self.specSensor.command('Acquisition.Start')

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
        """
        Updated: enqueue FULL hyperspectral line (WIDTH, bands).
        RGB conversion happens in _drain_and_update_plot (UI thread).
        """
        if self.acquisition_stopped==False:
            try:
                nbytes = int(nFrameSize)
                if nbytes <= 0:
                    return

                line16, bands = self._reshape_line_from_bytes(pBuffer, nbytes)
                if line16 is None:
                    return

                self._bands_in_line = bands

                # --- Determine current mode from globals ---
                # NOTE: these are global flags set by the worker thread


                if self.white_calibration_data:
                    mode = "white"
                elif self.dark_calibration_data:
                    mode = "dark"
                else:
                    mode = "scan"
                #print(mode)
                # If mode changed since last callback, finalize previous calibration
                if self._prev_cal_mode is None:
                    self._prev_cal_mode = mode
                elif mode != self._prev_cal_mode:
                    if self._prev_cal_mode == "white":
                        self._finalize_and_save_calibration("white")
                    elif self._prev_cal_mode == "dark":
                        self._finalize_and_save_calibration("dark")
                    self._prev_cal_mode = mode

                # Collect based on mode
                if mode == "white":
                    self.white_cal_lines.append(line16.copy())
                    # still show something on preview (optional)
                    line_for_display = line16

                elif mode == "dark":
                    self.dark_cal_lines.append(line16.copy())
                    line_for_display = line16

                else:
                    # scan data
                    # Apply ELM to scan lines if refs exist (store calibrated or raw as you prefer)
                    #white_cal_lines=np.asarray(self.dark_cal_lines)
                    #self.white_ref = white_cal_lines.mean(axis=(0, 1)).astype(np.float32)

                    #dark_cal_lines = np.asarray(self.dark_cal_lines)
                    #self.white_ref = dark_cal_lines.mean(axis=(0, 1)).astype(np.float32)
                    if self.apply_calibration:
                        out = self._apply_elm(line16)
                        if out.dtype == np.uint16 or out.dtype == np.uint32:
                            # refs missing -> still raw
                            self.hsi_lines.append(line16.copy())  # store raw (or skip storing)
                            line_for_display = line16
                        else:
                            # calibrated reflectance
                            self.hsi_lines.append(out.copy())  # store calibrated
                            line_for_display = out
                    else:
                        self.hsi_lines.append(line16.copy())
                        line_for_display = line16

                # enqueue COPY so safe after SDK returns
                try:
                    self.line_queue.put_nowait(line_for_display.copy())  # (WIDTH, bands)
                except Exception:
                    # drop oldest and retry
                    try:
                        self.line_queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        self.line_queue.put_nowait(line_for_display.copy())
                    except Exception:
                        pass

                self._lines_rcvd += 1
                if self._lines_rcvd == self.line_count:
                    self.acquisition_stopped=True
            except Exception:
                return


    def rgb_preview_from_reflectance_line(self,
            refl_line: np.ndarray,  # (WIDTH, bands) float32, assumed 0..1
            rgb_bands_0: np.ndarray,  # (3,) int, 0-based [R,G,B]
            brightness: float = 1.0,
    ) -> np.ndarray:
        """
        Stable RGB for ELM-calibrated reflectance lines.
        No per-line normalization. Assumes refl is already in [0,1].
        """
        rgb = refl_line[:, rgb_bands_0].astype(np.float32, copy=False)  # (W,3)
        rgb = brightness * rgb
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255.0 + 0.5).astype(np.uint8)



    def _drain_and_update_plot(self):
        # if self.acquisition_stopped:
            # print(self.acquisition_stopped)
        if self.acquisition_stopped==False:
            drained = 0
            if not hasattr(self, "_write_row"):
                self._write_row = 0

            # cache rgb band indices once
            if (self.lambda_vec is not None) and (self._rgb_band_idxs is None):
                if len(self.lambda_vec) == (self._bands_in_line or len(self.lambda_vec)):
                    self._rgb_band_idxs = find_rgb_bands(self.lambda_vec)

            while True:
                try:
                    line = self.line_queue.get_nowait()  # can be uint16 raw OR float32 calibrated
                except Empty:
                    break

                # If you queued calibrated lines when apply_calibration=True, then line is reflectance already.
                # If not calibrated, you can either:
                #   a) show raw with percentile stretch (your old way), OR
                #   b) apply ELM here for preview only.
                if line.dtype != np.uint16 and line.dtype != np.uint32:
                    # assume calibrated reflectance float line: (WIDTH,bands) in [0,1]
                    if self._rgb_band_idxs is not None:
                        rgb8 = self.rgb_preview_from_reflectance_line(
                            line, self._rgb_band_idxs, brightness=self.rgb_brightness
                        )
                    else:
                        # fallback if no wavelength vector
                        rgb16_3 = line[:, BAND_IDXS]
                        rgb8 = np.clip(rgb16_3 * 255.0, 0, 255).astype(np.uint8)
                else:
                    # raw path (no calibration): keep your old preview method
                    if BAND_IDXS.max() >= line.shape[1]:
                        continue
                    rgb16_3 = line[:, BAND_IDXS]
                    rgb8 = self._u16_to_u8(rgb16_3)

                self.rgb_img[self._write_row, :, :] = rgb8
                self._write_row = (self._write_row + 1) % ROLLING_HEIGHT
                drained += 1

            if drained:
                wr = self._write_row
                view = np.vstack((self.rgb_img[wr:], self.rgb_img[:wr]))
                self.im.set_data(view)
                self.ax.set_title(
                    f"RGB idxs {self._rgb_band_idxs.tolist() if self._rgb_band_idxs is not None else BAND_IDXS.tolist()} "
                    f"| lines: {self._lines_rcvd} bands: {self._bands_in_line} q: {self.line_queue.qsize()}"
                )
                self.canvas.draw_idle()

    def btnCameraConnect_clicked(self):
        self.btnCameraConnect.setEnabled(False)
        self.btnCameraDisconnect.setEnabled(True)

        QApplication.setOverrideCursor(
            Qt.WaitCursor)  # provide user feedback that we are waiting for a process to finish via cursor wait icon
        self._start_sensor_and_callback()
        QApplication.restoreOverrideCursor()

    def btnCameraDisconnect_clicked(self):
        self.btnCameraDisconnect.setEnabled(False)
        self.btnCameraConnect.setEnabled(True)

        # track camera connection
        self.cam_connected = False

    def updateSettingsToDevice(self):
        cam_frame_rate = str(getattr(rig_settings, "CAMERA_FRAME_RATE", ""))
        # cam_exp_time = str(getattr(rig_settings, "EXPOSURE_TIME", ""))
        cam_spat_bin = str(getattr(rig_settings, "CAMERA_SPATIAL_BIN", ""))
        cam_spec_bin = str(getattr(rig_settings, "CAMERA_SPECTRAL_BIN", ""))

        self.btnCameraConnect.setEnabled(False)
        command_status, message = send_command('CONNECT')
        command_status, message = send_command('FRAME_RATE,' + cam_frame_rate)

        READOUT_TIME = self.specSensor.getfloat("Camera.Image.ReadoutTime")[0]
        print(f"readout: {READOUT_TIME}")
        # cam_exp_time = str((1 / (float(cam_frame_rate))) * 1000+READOUT_TIME)  # milliseconds
        cam_exp_time = (1000 / (float(cam_frame_rate))) - READOUT_TIME  # milliseconds
        print(f"cam exposure calc: {cam_exp_time}")
        # command_status, message = send_command('EXPOSURE_TIME' + cam_exp_time)
        self.specSensor.setfloat("Camera.ExposureTime", cam_exp_time)
        exp_time = self.specSensor.getfloat("Camera.ExposureTime")
        print(f"Exposure time: {exp_time}")

        # val=str(self.cmbSpectralBin.currentIndex())
        command_status, message = send_command(
            'SPECTRAL_BINNING,' + cam_spec_bin)  # str(self.cmbSpectralBin.currentIndex()))
        command_status, message = send_command(
            'SPATIAL_BINNING,' + cam_spat_bin)  # str(self.cmbSpatialBin.currentIndex()))
        
        self.specSensor.setbool("Camera.Preprocessing.Enabled", True)
        self.specSensor.setbool("Camera.AutoNUC", True)

        return message

    def btnApplyAdjust_clicked(self):
        # update the setting first
        self.update_settings()

        # setting get values
        message = self.updateSettingsToDevice()
        # Update speed based on FPS
        rig_settings.RIG_SPEED = self.calcSpeedFromFPS(rig_settings.CAMERA_FRAME_RATE)

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
            return

        self.btnStartAcquire.setEnabled(False)
        self.btnStopAcquire.setEnabled(True)
        self.load_latest_calibration()
        # NOTE: If user in on tab 0, then we will prefer the line count based end position calculation
        # 1 line is Approx. 0.125mm distance (wrong)
        # 1 line is Approx. 0.16369mm distance (wrong)
        # TODO: Line count and it's derived end position for the scan bed are only saved when user hits `update setting` button
        # TODO: get linecount from setting and not gui is better

        # calculated what 1 line is in mm based on the scanning setup
        height = rig_settings.RIG_CAM_HEIGHT_OFFSET + rig_settings.RIG_CAM_HEIGHT
        object_length = (0.6842 * height) + 21.368
        actual_pitch = object_length / WIDTH
        # LINE_PITCH = (2 * height * math.tan(math.radians(CAMERA_LENSE_FOV) / 2)) / WIDTH
        print(f"calculated line_pitch from setup: {LINE_PITCH}")
        # 25mm additional offset is added to make sure we get the request lines as a just in case (25mm = ~200 line)
        # line_count_end_pos = rig_settings.RIG_BED_START + (int(self.textEditFrameCount.toPlainText()) * LINE_PITCH) # + 25
        line_count_end_pos = rig_settings.RIG_BED_START + (int(self.textEditFrameCount.toPlainText()) * actual_pitch) # + 25
        print(f"end pre-round: {line_count_end_pos}")
        line_count_end_pos = round(line_count_end_pos, 2)
        print(f"end post-round: {line_count_end_pos}")

        # check line count is valid
        if line_count_end_pos < 0:
            self.btnStartAcquire.setEnabled(True)
            self.btnStopAcquire.setEnabled(False)
            print(f"ERROR: end position calculated as less than 0.")
            return
        # check that we are not going past machine limits (max is 600mm)
        if line_count_end_pos > 600:
            # cap to max pos
            line_count_end_pos = 600

        rig_settings.RIG_BED_END = line_count_end_pos
        self.acquisition_stopped=False

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
        self.acquisition_stopped = True
        self.specSensor.command('Acquisition.Stop')
        print("Status: Scan completed")

        # finalize any pending calibration capture before saving scan
        try:
            if self._prev_cal_mode == "white":
                self._finalize_and_save_calibration("white")
            elif self._prev_cal_mode == "dark":
                self._finalize_and_save_calibration("dark")
            self._prev_cal_mode = "scan"
        except Exception:
            pass


        try:
            if len(self.hsi_lines) == 0:
                print("No hyperspectral lines collected; nothing to save.")
                return

            cube = np.stack(self.hsi_lines, axis=0)  # (lines, samples=WIDTH, bands)

            # get textbox for output folder dir
            rig_settings.OUTPUT_FOLDER = self.txtOutputFolderPath.text() # TODO: Make this so that the path is got from the saved config file, which is also needs updated via the gui in future development
            out_dir = getattr(rig_settings, "OUTPUT_FOLDER", ".")
            # out_dir = "C:\work\data"
            print(out_dir)
            os.makedirs(out_dir, exist_ok=True)

            # choose a filename (timestamped)
            ts = time.strftime("%Y%m%d_%H%M%S")
            base_path = os.path.join(out_dir, f"scan_{ts}")

            # wavelength vector must match bands
            wavelength = self.lambda_vec
            if wavelength is not None and len(wavelength) != cube.shape[2]:
                print(f"Warning: lambda_vec length {len(wavelength)} != bands {cube.shape[2]}. "
                      f"Saving without wavelength.")
                wavelength = None

            if self.chkSaveToFile.isChecked():
                data_path, hdr_path = write_envi_bil(
                    base_path=base_path,
                    cube=cube.astype(np.float32, copy=False),  # typical sensor output
                    wavelength=wavelength,
                    description="University of Lincoln, Linescan HSI image (BIL)",
                    extra_header={
                        "sensor type": "SpecSensor",
                        "camera_frame_rate": getattr(rig_settings, "CAMERA_FRAME_RATE", ""),
                        "exposure": getattr(rig_settings, "CAMERA_EXPOSURE", ""),
                    }
                )
                print(f"Saved ENVI: {data_path} and {hdr_path}")

            # Explicitly drop large buffers ASAP
            try:
                del cube
            except Exception:
                pass

            self._clear_memory_after_save(aggressive=True)


        except Exception as e:
            print(f"ENVI save failed: {e}")
        finally:
            # clear buffer for next scan
            self.hsi_lines = []


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
        # Save current settings to YAML on exit
        try:
            # attempt to persist UI values (camera + rig)
            self.update_settings()
        except Exception:
            pass

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
                 rig_controller: RIGController = None,  # if None, then use the one in main_window
                 cam_height: float = 0.0,
                 scan_speed: float = 443.33,
                 init_pos: float = 80.0,
                 end_scan_pos: float = 650.0
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
        self.main_window.is_running = True

        # Pause control of the routine for getting user input when calibration is enabled
        self._paused = False
        self._pause_mutex = QMutex()
        self._pause_condition = QWaitCondition()
        self._user_cancelled = False

    def stop(self):
        """Stop the scan routine"""
        self._is_running = False
        self.main_window.is_running = False

        # if needed, wake the thread in case it’s waiting
        with QMutexLocker(self._pause_mutex):
            self._paused = False
            self._pause_condition.wakeAll()

        # check if we still have a controller that is connected first
        if self.rig_controller is not None and self.rig_controller.is_connected():
            self.rig_controller.emergency_stop()  # software e-stop the controller
            self.rig_controller.disconnect()  # disconnect from the controller

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

    def home(self):
        # Step 1: Reset and home
        self.status_update.emit("Resetting controller and homing axes...")
        self.rig_controller.reset_controller()
        time.sleep(1)

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

    def run(self):
        """This runs the scan routine on in a separate thread"""
        try:
            # rig_settings.RIG_TIMEOUT_READ_ONLY = 95
            # rig_settings.RIG_TRAVEL_SPEED_READ_ONLY = 6000

            self.status_update.emit("Starting scan routine...")

            if not self.rig_controller.serial_conn or not self.rig_controller.serial_conn.is_open:
                self.error_occurred.emit("ERROR: Not connected to controller!")
                return

            if not self.main_window.chkSkipHoming.isChecked():
                self.home()
            else:
                # Pause: dialog box here to get user confirmation that they understand the risk of skipping homing
                self.status_update.emit("skipping homing. Waiting for user to acknoledge risk...")

                if not self.wait_for_confirmation("Skipping homing is risky, as it could result in serious damage to the machine and/or scanning subject if it has not been home prior or has moved since last homing. Do you acknowledge the risk and wish to continue?"):
                    self.status_update.emit("Scan cancelled by user.")
                    return


            if not self._is_running:
                return

            # Check to see if we need to move to calibration positions for black and white
            if self.main_window.chkCalibration.isChecked():
                print(f"Moving to white calibration strip")
                # NOTE: You may need to update the Y/Z positions for your strip in rig_settings.py if incorrect
                y_pos = rig_settings.RIG_WHITE_CAL_POS_READ_ONLY  # Position still needs calibrating
                z_pos = self.cam_height
                self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)  # Set feedrate in mm/min

                # Move to Y axis position (strip location)
                success_y = self.rig_controller.move_axis('Y', y_pos)
                success_z = self.rig_controller.move_axis('Z', z_pos)

                # Wait for moving with interruptible sleep
                print("Waiting for move to white calibration strip to complete...")
                start_time = time.time()
                while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY:
                    pos = self.rig_controller.get_current_position()
                    if pos is not None and pos["Y"] == rig_settings.RIG_WHITE_CAL_POS_READ_ONLY and pos["Z"] == z_pos:
                        break
                    self.msleep(100)  # Use QThread's msleep for better integration

                if success_y and success_z:
                    print("Status: At white calibration strip")
                else:
                    print("Status: Move failed")

                # Pause: dialog box here to get user confirmation to continue routine
                self.status_update.emit("At white calibration position. Waiting for user confirmation...")

                if not self.wait_for_confirmation("Calibration paused at white strip. Do you want to continue?"):
                    self.status_update.emit("Scan cancelled by user.")
                    return

                self.main_window.white_calibration_data = True
                self.main_window.dark_calibration_data = False
                # Start acquisition
                self.status_update.emit("Starting camera acquisition...")
                self.msleep(2000)
                self.main_window.specSensor.command('Acquisition.Start')
                self.msleep(2000)
                self.main_window.specSensor.command('Acquisition.Stop')

                self.status_update.emit("Continuing scan routine...")

                print(f"Moving to black calibration strip")

                # NOTE: You may need to update the Y/Z positions for your strip in rig_settings.py if incorrect
                y_pos = rig_settings.RIG_BLACK_CAL_POS_READ_ONLY  # Position still needs calibrating
                z_pos = self.cam_height
                self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)  # Set feedrate in mm/min

                # Move to Y axis position (strip location)
                success_y = self.rig_controller.move_axis('Y', y_pos)
                success_z = self.rig_controller.move_axis('Z', z_pos)

                # Wait for moving with interruptible sleep
                print("Waiting for move to black calibration strip to complete...")
                start_time = time.time()
                while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY:
                    pos = self.rig_controller.get_current_position()
                    if pos is not None and pos["Y"] == rig_settings.RIG_BLACK_CAL_POS_READ_ONLY and pos["Z"] == z_pos:
                        break
                    self.msleep(100)  # Use QThread's msleep for better integration

                if success_y:
                    print("Status: At black calibration strip")
                else:
                    print("Status: Move failed")
                # Pause: dialog box here to get user confirmation to continue routine
                self.status_update.emit("At black calibration position. Waiting for user confirmation...")

                if not self.wait_for_confirmation("Calibration paused at black strip. Do you want to continue?"):
                    self.status_update.emit("Scan cancelled by user.")
                    return
                self.main_window.specSensor.setbool("Camera.Shutter.IsToggle", False)

                self.main_window.white_calibration_data = False
                self.main_window.dark_calibration_data = True

                self.status_update.emit("Starting camera acquisition...")
                self.msleep(2000)
                self.main_window.specSensor.command('Acquisition.Start')
                self.msleep(2000)
                self.main_window.specSensor.command('Acquisition.Stop')

                self.main_window.white_calibration_data = False
                self.main_window.dark_calibration_data = False
                self.main_window.specSensor.setbool("Camera.Shutter.IsToggle", True)

                self.status_update.emit("Continuing scan routine...")

                if not self._is_running:
                    return

            # Move to initial position
            self.status_update.emit("Moving to initial position...")
            self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)
            self.rig_controller.move_axis("Y", self.init_pos)
            self.rig_controller.move_axis('Z', self.cam_height)

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
                # print(f'end postion move - current Pos Y: {pos["Y"]}')
                if pos is not None and pos["Y"] >= self.scan_pos and pos["Z"] == self.cam_height:
                    break
                self.msleep(100)

            if not self._is_running:
                return

            self.status_update.emit("Scan routine completed successfully!")
            self.scan_complete.emit()
            # self.msleep(1000)

            # Return to initial position
            self.status_update.emit("Returning to initial position...")
            self.rig_controller.set_feed_rate(rig_settings.RIG_TRAVEL_SPEED_READ_ONLY)
            self.rig_controller.move_axis("Y", self.init_pos)
            self.rig_controller.move_axis('Z', self.cam_height)

            start_time = time.time()
            while time.time() - start_time < rig_settings.RIG_TIMEOUT_READ_ONLY and self._is_running:
                pos = self.rig_controller.get_current_position()
                if pos is not None and pos["Y"] == self.init_pos and pos["Z"] == self.cam_height:
                    break
                self.msleep(100)


        except Exception as e:
            self.error_occurred.emit(f"Error during scan: {str(e)}")

