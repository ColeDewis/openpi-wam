import cv2
from harvesters.core import Harvester
import threading

def buffer_to_image(buffer):
    component = buffer.payload.components[0]
    width = component.width
    height = component.height
    data_format = component.data_format
    
    # Harvesters buffers are read-only memory blocks. 
    # OpenCV hates read-only memory. We MUST .copy() it.
    raw_data = component.data.copy()

    # 1. Handle Monochrome
    if "Mono" in data_format:
        return raw_data.reshape((height, width))

    # 2. Handle RGB / BGR Formats
    elif "RGB" in data_format or "BGR" in data_format:
        # Determine bytes per pixel (e.g., RGB8 is 3, RGBa8 is 4)
        channels = 4 if "a" in data_format.lower() else 3
        img_nd = raw_data.reshape((height, width, channels))
        
        # If the camera explicitly sends RGB, flip it to BGR for OpenCV
        if "RGB" in data_format:
            # Slicing the first 3 channels reverses RGB to BGR, 
            # and leaves the Alpha channel (if it exists) alone or ignores it
            img_nd = img_nd[:, :, :3][:, :, ::-1]
            
        return img_nd

    # 3. Handle Bayer Formats (Standard for FLIR Color Sensors)
    elif "Bayer" in data_format:
        img_1d = raw_data.reshape((height, width))
        
        # If your image is blue, your camera's actual Bayer pattern is swapped.
        # We explicitly map the conversions to output BGR to fix the tint.
        if "BayerRG" in data_format:
            # FLIR's "RG" often actually requires the "BG" conversion to look correct in OpenCV
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerBG2BGR)
        elif "BayerBG" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerRG2BGR)
        elif "BayerGB" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerGR2BGR)
        elif "BayerGR" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerGB2BGR)
        else:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerBG2BGR)

    # 4. Catch-all
    else:
        raise ValueError(f"Can't convert {data_format}")

class FLIRStream:
    def __init__(self, device):
        self.device = device
        self.latest_image = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Starts the camera and the background fetching thread."""
        self.running = True
        self.device.start()
        
        # Daemon=True means this thread will auto-kill if your main script crashes
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Background loop constantly pulling the newest frame."""
        while self.running:
            with self.device.fetch(timeout=1) as buffer:
                img = buffer_to_image(buffer)
                
                # Lock the variable just long enough to update it 
                # so we don't read a half-written frame in the main thread
                with self.lock:
                    self.latest_image = img


    def read(self):
        """Returns the most recent frame instantly."""
        with self.lock:
            if self.latest_image is not None:
                # Return a copy so your main loop can modify it safely
                return self.latest_image.copy()
            return None

    def stop(self):
        """Cleanly shuts down the thread and camera."""
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.device.stop()

def open_flir(harvester: Harvester, serial_num: str = None):
    if serial_num:
        device = harvester.create({'serial_number': str(serial_num)})
    else:
        device = harvester.create()
        
    device.start()
    return device

# TODO: set up script to cleanly read two of these now, format for model, and test if I can do the dummy inference w the images. If that works, can try adding
# real robot joints after.

# WRIST FLIR: 18475176
# OTHER FLIR: 18475182
h = Harvester()
h.add_file("/opt/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti")
h.update()

wrist_cam = open_flir("18475176")
scene_cam = open_flir("18475182")
wrist_stream = FLIRStream(wrist_cam)
scene_stream = FLIRStream(scene_cam)

try:
    while True:
        with device.fetch(timeout=3) as buffer:
            img = buffer_to_image(buffer)
            cv2.imshow("flir", img)
            
            # Wait 1ms and check if the pressed key is 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Closing video feed...")
                break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # This guarantees the camera and windows close cleanly 
    # whether you press 'q' OR use Ctrl+C
    print("Cleaning up...")
    device.stop()
    device.destroy()
    h.reset()
    cv2.destroyAllWindows()
