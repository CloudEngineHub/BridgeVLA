import pyzed.sl as sl
import numpy as np
import cv2
import threading
from scipy.spatial.transform import Rotation as R
import time
import os

def save_rgb_image(rgb_array, save_path):
    """
    Save observation["3rd"]["rgb"] to the given path.
    :param rgb_array: numpy array, HxWx3, RGB, values in [0,255] or [0,1]
    :param save_path: str, destination path
    """
    # float input in [0,1] is converted to uint8 [0,255] first
    if rgb_array.dtype != np.uint8:
        rgb_array = (rgb_array * 255).clip(0, 255).astype(np.uint8)
    # OpenCV writes BGR
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    # Make sure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr)    

def get_cam_extrinsic(type, which='left'):
    if type == "3rd":
        if which == 'right':
            transform = [[-0.00340802 , 0.40362924 ,-0.91491629 , 0.64273863],
                        [ 0.99011803 ,-0.12690632 ,-0.05967479 ,-0.50882022],
                        [-0.14019515 ,-0.90607848 ,-0.39920809,  0.74841034],
                        [ 0.       ,   0.    ,      0.    ,      1.        ]]
        elif which == 'left':
            transform = [[ 0.79176777,  0.0224346,   0.6104101,  -1.2317986 ],
                        [-0.03084025, -0.99658246,  0.07663085, -0.01006371],
                        [ 0.61004318, -0.07949904, -0.78836998,  0.7087538 ],
                        [ 0.,          0.,          0.,          1.        ]]



    elif type == "wrist":
        trans = np.array([0.6871684912377796, -0.7279882263970943, 0.8123566411202088])
        quat = np.array([-0.869967706085017, -0.2561670369853595, 0.13940123346877276, 0.39762034107764127])
    else:
        raise ValueError("Invalid type")

    return np.array(transform)
 
class ZedCam:
    def __init__(self,serial_number, resolution=None): # resolution=(480, 640)
        self.zed = sl.Camera()
        self.init_zed(serial_number=serial_number)
        
        if resolution:
            self.img_size = sl.Resolution()
            self.img_size.height = resolution[0]
            self.img_size.width = resolution[1]
        else:
            self.img_size = self.zed.get_camera_information().camera_configuration.resolution
            
        self.center_crop = False
        self.center_crop_size = (480, 640)
        

    def init_zed(self,serial_number):
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(serial_number)
        init_params.camera_resolution = sl.RESOLUTION.HD1080 # sl.RESOLUTION.AUTO, sl.RESOLUTION.HD720, sl.RESOLUTION.HD1080
        
        init_params.camera_fps = 30  # Set fps at 30
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL # Use ULTRA depth mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER # Use millimeter units (for depth measurements)

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(err)+". Exit program.")
            exit()
            
        # Init 50 frames
        image = sl.Mat()
        for _ in range(50):
            runtime_parameters = sl.RuntimeParameters()
            # Grab an image, a RuntimeParameters object must be given to grab()
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # A new image is available if grab() returns SUCCESS
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                
    def capture(self):
        image = sl.Mat(self.img_size.width, self.img_size.height, sl.MAT_TYPE.U8_C4)
        depth_map = sl.Mat(self.img_size.width, self.img_size.height, sl.MAT_TYPE.U8_C4)
        point_cloud = sl.Mat()

        while True:
            runtime_parameters = sl.RuntimeParameters()
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
                # A new image and depth is available if grab() returns SUCCESS
                self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, self.img_size) # Retrieve left image
                self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU, self.img_size) # Retrieve depth
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.img_size)
                frame_timestamp_ms = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_microseconds()
                break
            
        rgb_image = image.get_data()[..., :3] 
        depth = depth_map.get_data()
        depth[np.isnan(depth)] = 0
        depth_image_meters = depth * 0.001
        pcd = point_cloud.get_data()
        pcd[np.isnan(pcd)] = 0
        pcd = pcd[..., :3] * 0.001
        
        if self.center_crop:
            result_dict = {
                "rgb": self.center_crop_img(rgb_image),
                "depth": self.center_crop_img(depth_image_meters),
                "pcd": self.center_crop_img(pcd),
                "timestamp_ms": frame_timestamp_ms / 1000.0,
            }
        else:
            result_dict = {
                "rgb": rgb_image,
                "depth": depth_image_meters,
                "pcd": pcd,
                "timestamp_ms": frame_timestamp_ms / 1000.0,
            }
        return result_dict

    def capture_rgb(self):
        """Lightweight grab: left-eye RGB only, no depth or point cloud, for high-frame-rate recording during motion.

        grab() blocks until the camera delivers the next frame, so calling this in a loop naturally runs at the
        camera's configured fps (currently HD1080 @ 30fps) with no extra pacing needed.
        Disabling enable_depth skips the NEURAL depth computation, which 30fps may otherwise not keep up with.

        Returns:
            (frame_bgr, cam_ts_ms):
                frame_bgr is an (H, W, 3) uint8 BGR standalone copy (ready for cv2.imwrite);
                cam_ts_ms is that frame's camera timestamp in milliseconds.
            Returns (None, None) when the grab fails.
        """
        if not hasattr(self, "_rgb_mat"):
            self._rgb_mat = sl.Mat(
                self.img_size.width, self.img_size.height, sl.MAT_TYPE.U8_C4
            )
        runtime_parameters = sl.RuntimeParameters()
        try:
            runtime_parameters.enable_depth = False
        except AttributeError:
            pass  # older SDK without the switch — depth just costs extra time
        if self.zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
            return None, None
        self.zed.retrieve_image(self._rgb_mat, sl.VIEW.LEFT, sl.MEM.CPU, self.img_size)
        try:
            cam_ts_ms = float(
                self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
            )
        except Exception:
            cam_ts_ms = time.time() * 1000.0
        frame_bgr = self._rgb_mat.get_data()[..., :3].copy()
        return frame_bgr, cam_ts_ms



    def center_crop_img(self, img):
        if len(img.shape) == 2:
            crop_img = np.zeros((self.center_crop_size[0], self.center_crop_size[1]), dtype=img.dtype)
            crop_img = img[(img.shape[0] - self.center_crop_size[0]) // 2: (img.shape[0] + self.center_crop_size[0]) // 2,
                          (img.shape[1] - self.center_crop_size[1]) // 2: (img.shape[1] + self.center_crop_size[1]) // 2]
            return crop_img
        else:
            channel = img.shape[-1]
            crop_img = np.zeros((self.center_crop_size[0], self.center_crop_size[1], channel), dtype=img.dtype)
            crop_img = img[(img.shape[0] - self.center_crop_size[0]) // 2: (img.shape[0] + self.center_crop_size[0]) // 2,
                            (img.shape[1] - self.center_crop_size[1]) // 2: (img.shape[1] + self.center_crop_size[1]) // 2]
        return crop_img
        
    
    def stop(self):
        # Close the camera
        self.zed.close()
        

class Camera:
    def __init__(self, camera_type="all", timestamp_tolerance_ms=80):
        # ZED serial numbers; 0 = open the first camera the SDK reports. Set
        # both explicitly when more than one ZED is connected:
        #   ZED_3RD_SERIAL=<serial> ZED_WRIST_SERIAL=<serial> python ...
        static_serial_number = int(os.environ.get("ZED_3RD_SERIAL", 0))
        wrist_serial_number = int(os.environ.get("ZED_WRIST_SERIAL", 0))

        if camera_type == "all":
            self.cams =  [ZedCam(serial_number= static_serial_number ), ZedCam(serial_number=wrist_serial_number)]
            self.camera_types = ["3rd", "wrist"]

        elif camera_type == "3rd":
            self.cams = [ZedCam(serial_number= static_serial_number )]
            self.camera_types = ["3rd"]

        elif camera_type == "wrist":
            self.cams = [ZedCam(serial_number=wrist_serial_number)]
            self.camera_types = ["wrist"]
        
        else:
            raise ValueError("Invalid camera type, please choose from 'all', '3rd', 'wrist'")
        
        self.timestamp_tolerance_ms = timestamp_tolerance_ms
        
        
    def _capture_frame(self, idx, result_dict, start_barrier, done_barrier):
        """
        start_barrier: A threading.Barrier to ensure all threads start capturing at the same time.
        done_barrier: A threading.Barrier to ensure all threads finish capturing before main thread proceeds.
        """
        cam = self.cams[idx]
        camera_type = self.camera_types[idx]
        # Wait here until all threads are ready (software-level synchronization)
        start_barrier.wait()
        result_dict[camera_type] = cam.capture()
        # Signal that this thread is done
        done_barrier.wait()
        
    def capture_frames_multi_thread(self):
        result_dict = {}
        if len(self.cams) == 1:
            result_dict[self.camera_types[0]] = self.cams[0].capture()
            _ = [result_dict[cam].pop("timestamp_ms", None) for cam in result_dict] # remove timestamps
            return result_dict
        
        else:
            num_cameras = len(self.cams)

            # Two barriers: one to synchronize the start, one to wait for all threads to finish
            start_barrier = threading.Barrier(num_cameras)
            done_barrier = threading.Barrier(num_cameras)

            threads = []

            for idx in range(num_cameras):
                t = threading.Thread(
                    target=self._capture_frame,
                    args=(idx, result_dict, start_barrier, done_barrier)
                )
                threads.append(t)
                t.start()

            # Wait for all threads to finish
            for t in threads:
                t.join()

            # Timestamp alignment step
            # 1) Gather all timestamps
            timestamps = [result_dict[cam]["timestamp_ms"] for cam in result_dict]
            _ = [result_dict[cam].pop("timestamp_ms", None) for cam in result_dict] # remove timestamps
            
            # 2) Compute min, max, and check difference
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            diff_ts = max_ts - min_ts  # in ms

            # 3) Compare difference with the tolerance
            if diff_ts > self.timestamp_tolerance_ms:
                print("Timestamps are not aligned, difference is", diff_ts, "ms,", "discard frames")
                return None
            else:
                return result_dict
    
    
    def capture(self):
        while True:
            result_dict = self.capture_frames_multi_thread()
            if result_dict is not None:
                break
        return result_dict
    
    
    def stop(self):
        for cam in self.cams:
            cam.stop()

    def capture_for_collection(self):
        """Capture data in format compatible with data_collection_main_single_display_cycle_share_camera.py"""
        if "3rd" not in self.camera_types:
            raise ValueError("3rd camera not initialized")
            
        result = self.cams[self.camera_types.index("3rd")].capture()
        return {
            "rgb": result["rgb"],
            "depth": result["depth"],
            "pcd": result["pcd"]
        }



           
            
if __name__ == "__main__":
    cameras = Camera(camera_type="3rd")

    import open3d as o3d
    observation = {}
    
    observation=cameras.capture()
    observation["3rd"]["rgb"]=observation["3rd"]["rgb"][:,:,::-1].copy()
    
    # Why it must be written this way rather than passing observation["3rd"]["rgb"][:,:,::-1].copy() to is_pcd:
    # the reversed view breaks contiguity, so a later reshape would produce misaligned data.
    def convert_pcd_to_base(
            type="3rd",
            pcd=[]
        ):
        transform = get_cam_extrinsic(type)
        
        h, w = pcd.shape[:2]
        pcd = pcd.reshape(-1, 3)
        
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
        pcd = (transform @ pcd.T).T[:, :3]
        
        pcd = pcd.reshape(h, w, 3)
        return pcd 
    
    def vis_pcd(pcd, rgb):
        # Reshape the point cloud and colours to (N, 3)
        pcd_flat = pcd.reshape(-1, 3)  # (200 * 200, 3)
        rgb_flat = rgb.reshape(-1, 3) / 255.0  # (200 * 200, 3)

        # Write the points and colours out as a PLY file
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_flat)  # point positions
        pcd.colors = o3d.utility.Vector3dVector(rgb_flat)  # matching colours
        o3d.visualization.draw_geometries([pcd])

    observation["3rd"]["pcd"] = convert_pcd_to_base("3rd", observation["3rd"]["pcd"])
    vis_pcd(observation["3rd"]["pcd"], observation["3rd"]["rgb"])
