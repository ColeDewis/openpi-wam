import ctypes
from multiprocessing import Event
from multiprocessing import Lock
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Value
from multiprocessing import shared_memory
import time

from harvesters.core import Harvester

try:
    from genicam.gentl import TimeoutException
except ImportError:
    from harvesters.core import TimeoutException
import numpy as np
from termcolor import cprint


def _convert_frame(raw, width, height, data_format):
    """Runs in the child process only."""
    if "Mono" in data_format:
        return raw.reshape((height, width))
    if "RGB" in data_format or "BGR" in data_format:
        channels = 4 if "a" in data_format.lower() else 3
        img = raw.reshape((height, width, channels))
        if "RGB" in data_format:
            img = img[:, :, :3][:, :, ::-1]
        return img
    if "Bayer" in data_format:
        import cv2

        img_1d = raw.reshape((height, width))
        if "BayerRG" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerBG2BGR)
        if "BayerBG" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerRG2BGR)
        if "BayerGB" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerGR2BGR)
        if "BayerGR" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerGB2BGR)
        return cv2.cvtColor(img_1d, cv2.COLOR_BayerBG2BGR)
    raise ValueError(f"Can't convert {data_format}")


def _configure_camera(device, serial, camera_settings, error_queue):
    """Use SpinView to tweak camera settings."""
    if not camera_settings:
        return
    try:
        nm = device.remote_device.node_map

        exposure_time = camera_settings.get("exposure_time")
        gain = camera_settings.get("gain")
        white_balance = camera_settings.get("white_balance")
        fps = camera_settings.get("fps")

        if exposure_time is not None:
            nm.ExposureAuto.value = "Off"
            nm.ExposureTime.value = float(exposure_time)

        if gain is not None:
            nm.GainAuto.value = "Off"
            nm.Gain.value = float(gain)

        if fps is not None:
            nm.AcquisitionFrameRateEnable.value = True
            nm.AcquisitionFrameRate.value = float(fps)

        if white_balance is None:
            nm.BalanceWhiteAuto.value = "Continuous"
        else:
            nm.BalanceWhiteAuto.value = "Off"
            if isinstance(white_balance, list | tuple) and len(white_balance) == 2:
                r, b = white_balance
                nm.BalanceRatioSelector.value = "Red"
                nm.BalanceRatio.value = float(r)
                nm.BalanceRatioSelector.value = "Blue"
                nm.BalanceRatio.value = float(b)
    except Exception as e:
        error_queue.put(f"[{serial}] Warning setting config: {e}")


def _flir_worker(
    serial,
    cti_path,
    startup_queue,
    frame_id,
    timestamp_ns,
    consumed,
    lock,
    running_event,
    error_queue,
    camera_settings=None,
):
    """Own the camera in its own process and publish frames via shared memory."""
    device = None
    shm = None
    shm_array = None

    try:
        harvester = Harvester()
        harvester.add_cti_file(cti_path)
        harvester.update_device_info_list()
        device = harvester.create_image_acquirer(serial_number=str(serial))
        _configure_camera(device, serial, camera_settings, error_queue)
        device.start_image_acquisition()

        while running_event.is_set():
            try:
                with device.fetch_buffer(timeout=0.5) as buffer:
                    if len(buffer.payload.components) == 0:
                        continue

                    with lock:
                        need_publish = consumed.value or frame_id.value == 0
                    if not need_publish:
                        continue

                    component = buffer.payload.components[0]
                    raw = component.data.copy()
                    img = _convert_frame(
                        raw,
                        component.width,
                        component.height,
                        component.data_format,
                    )

                    if shm is None:
                        shm = shared_memory.SharedMemory(create=True, size=img.nbytes)
                        shm_array = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
                        startup_queue.put(
                            {
                                "shm_name": shm.name,
                                "shape": img.shape,
                                "dtype": str(img.dtype),
                            }
                        )

                    with lock:
                        shm_array[:] = img
                        frame_id.value += 1
                        timestamp_ns.value = time.time_ns()
                        consumed.value = False

            except TimeoutException:
                continue
            except Exception as e:
                if not running_event.is_set():
                    break
                error_queue.put(str(e))
                time.sleep(0.05)

    except Exception as e:
        error_queue.put(f"Fatal init error on {serial}: {e}")
    finally:
        try:
            if device is not None:
                device.stop_image_acquisition()
                device.destroy()
        except Exception:
            pass
        if shm is not None:
            shm.close()
            shm.unlink()


class FLIRProcessStream:
    """Process-backed FLIR stream: capture and Bayer conversion never block the parent."""

    def __init__(
        self,
        serial_num,
        cti_path="/opt/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti",
        camera_settings: dict | None = None,
    ):
        self.serial_num = serial_num
        self.cti_path = cti_path
        self.camera_settings = camera_settings

        self.frame_id = Value(ctypes.c_uint64, 0)
        self.timestamp_ns = Value(ctypes.c_uint64, 0)
        self.consumed = Value(ctypes.c_bool, 1)
        self.lock = Lock()
        self.running_event = Event()
        self.startup_queue = Queue()
        self.error_queue = Queue()

        self.process = None
        self.shm = None
        self.shm_array = None

    def start(self):
        self.running_event.set()
        self.process = Process(
            target=_flir_worker,
            args=(
                self.serial_num,
                self.cti_path,
                self.startup_queue,
                self.frame_id,
                self.timestamp_ns,
                self.consumed,
                self.lock,
                self.running_event,
                self.error_queue,
                self.camera_settings,
            ),
            daemon=True,
        )
        self.process.start()

        # Wait for the child to publish its shared-memory handle after its
        # first successfully converted frame.
        info = self.startup_queue.get(timeout=10.0)
        self.shm = shared_memory.SharedMemory(name=info["shm_name"])
        self.shm_array = np.ndarray(
            info["shape"],
            dtype=np.dtype(info["dtype"]),
            buffer=self.shm.buf,
        )
        cprint(
            f"[FLIRProcessStream] {self.serial_num} attached to shm "
            f"'{info['shm_name']}' shape={info['shape']}",
            "green",
        )

    def read(self, last_seen_id):
        with self.lock:
            current_id = self.frame_id.value
            if current_id in (last_seen_id, 0):
                return None, last_seen_id, None
            img = self.shm_array.copy()
            timestamp_ns = self.timestamp_ns.value
            self.consumed.value = True
        return img, current_id, timestamp_ns

    def stop(self):
        self.running_event.clear()
        if self.process is not None:
            self.process.join(timeout=3.0)
            if self.process.is_alive():
                cprint(
                    f"WARNING: FLIR process {self.serial_num} did not exit, terminating.",
                    "red",
                )
                self.process.terminate()
        if self.shm is not None:
            self.shm.close()


class MultiFLIRManager:
    def __init__(
        self,
        camera_configs: dict,
        gentl_path: str = "/opt/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti",
    ):
        """camera_configs maps names to dicts like ``{"serial": "...", "fps": 30}``."""
        cprint("Initializing FLIR camera processes...", "green")
        self.streams = {}
        self.last_ids = {}
        for name, cfg in camera_configs.items():
            camera_settings = {k: v for k, v in cfg.items() if k != "serial"}
            self.streams[name] = FLIRProcessStream(
                cfg["serial"],
                gentl_path,
                camera_settings=camera_settings,
            )
            self.last_ids[name] = 0

    def start_all(self):
        cprint("Starting all FLIR camera processes...", "green")
        for stream in self.streams.values():
            stream.start()
        time.sleep(0.5)

    def read_all(self):
        """Return ``{name: (image, timestamp_ns)}`` for cameras with new frames."""
        new_frames = {}
        for name, stream in self.streams.items():
            img, new_id, timestamp_ns = stream.read(self.last_ids[name])
            if img is not None:
                self.last_ids[name] = new_id
                new_frames[name] = (img, timestamp_ns)
        return new_frames

    def stop_all(self):
        cprint("Shutting down FLIR camera processes...", "red")
        for stream in self.streams.values():
            stream.stop()
