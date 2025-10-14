
def _start_sensor_and_callback(self):
    self.specSensor = SpecSensor(sdkFolder='./libs/')
    profiles = self.specSensor.getProfiles()
    if not profiles:
        QtWidgets.QMessageBox.critical(self, "SpecSensor", "No devices found.")
        return

    # open first profile (or pick by name)
    err, _ = self.specSensor.open(profiles[16], autoInit=True)
    if err != 0:
        QtWidgets.QMessageBox.critical(self, "SpecSensor", f"Open failed: {err}")
        return

    # Keep strong reference so it won't be GC’ed
    self._callback_ref = self._onDataCallback
    self.specSensor.sensor.registerDataCallback(self._callback_ref)

    # start acquisition
    self.specSensor.command('Acquisition.Start')


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