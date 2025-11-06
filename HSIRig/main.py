import sys
from PyQt5.QtWidgets import QApplication
from views.main_window_view import MainWindow

def main():
    # Set environment variable for Qt platform
    #os.environ["QT_QPA_PLATFORM"] = "xcb"

    # Ensure that the QApplication is initialized correctly
    app = QApplication(sys.argv)

    main_window = MainWindow()
    # camera connection tab button event setup
    main_window.btnCameraConnect.clicked.connect(main_window.btnCameraConnect_clicked)
    main_window.btnCameraDisconnect.clicked.connect(main_window.btnCameraDisconnect_clicked)
    main_window.btnApplyAdjust.clicked.connect(main_window.btnApplyAdjust_clicked)
    main_window.btnStartAcquire.clicked.connect(main_window.btnStartAcquire_clicked)
    main_window.btnStopAcquire.clicked.connect(main_window.btnStopAcquire_clicked)
    main_window.btnRigConnect_2.clicked.connect(main_window.connect_controller)
    main_window.btnRigDisconnect_2.clicked.connect(main_window.disconnect_controller)

    main_window.apply_stylesheet()
    #main_window.camera_feed.plot_example()
    #camera_feed = MatplotlibWidget()
    #camera_feed = QtWidgets.QLabel(self.cameraFeedGroupBox)
    #camera_feed.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    #camera_feed.setObjectName("camera_feed")
    #horizontalLayout_2.addWidget(camera_feed)
    main_window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()