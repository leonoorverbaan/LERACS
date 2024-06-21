import os
import cv2
import cv2.aruco as aruco
import rospy
import numpy as np
#import pyzed.sl as sl

FRAMEWIDTH = 1920

# define names of ArUco tags supported by OpenCV.
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    from: https://stackoverflow.com/questions/75750177/solve-pnp-or-estimate-pose-single-markers-which-is-better
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    i = 0
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    print(f"detected:\n rvecs {rvecs} \ntvecs:{tvecs}")
    return np.asarray(rvecs), np.asarray(tvecs), np.asarray(trash)
    # return rvecs, tvecs, trash


class Camera:
    def __init__(self, camera_index=4
                 , name="default", marker_size=0.04):
        self.name = name
        self.video_source = camera_index
        self.parameters = aruco.DetectorParameters()
        self.marker_size = marker_size
        # open streaming camera
        self._init_streaming()
        # camera intrinsics (default is the left camera)
        self.cur_side = "right" # record the last call for the side of get_image
        self.camera_matrix_left = np.array([[1069.86, 0.0, 929.96],
                                            [0.0, 1069.8101, 540.947],
                                            [0.0, 0.0, 1.0]])
        self.dist_coeff_left = np.array([-0.0458, 0.0162, -0.0001, -0.0009, 0.0068])
        self.camera_matrix_right = np.array([[387.0155334472656, 0, 316.3547668457031],
                                             [0, 387.0155334472656, 243.0697021484375],
                                             [0, 0, 1]])
        self.dist_coeff_right = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self._load_calibration(self.cur_side)
        # store the image/depth/point cloud
        #self.img_mat = sl.Mat()
        # store the marker coordinates in pixels and in the world
        self.marker_pixel_coordinates = None
        self.detected_markers = dict()

    def _init_streaming(self):
        # Initialize video capture with the camera index
        #self.vid = cv2.VideoCapture(self.video_source)
        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_V4L2)
        #self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_GSTREAMER)

        if not self.vid.isOpened():
            rospy.logerr(f"Unable to open video source: {self.video_source}")
            raise IOError(f"Failed to open video source: {self.video_source}")

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # 7680  4320

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.vid.set(cv2.CAP_PROP_FOURCC, fourcc)

        rospy.loginfo(f"Camera {self.name} streaming from source {self.video_source}")

    def _load_calibration(self, side="right"):
        if side == "left":
            self.camera_matrix = self.camera_matrix_left
            self.dist_coeff = self.dist_coeff_left
        elif side == "right":
            self.camera_matrix = self.camera_matrix_right
            self.dist_coeff = self.dist_coeff_right
        else:
            rospy.logwarn("Invalid side specified for calibration loading. Defaulting to left.")
            self.camera_matrix = self.camera_matrix_left
            self.dist_coeff = self.dist_coeff_left

        rospy.loginfo(f"Loaded {side} camera calibration")

    def close(self):
        self.vid.release()

    def reset(self):
        import time
        self.close()
        time.sleep(0.3)
        self._init_streaming()
        time.sleep(1)

    def get_img(self, side: str = "right") -> np.ndarray:
        """Grab a frame from the video capture and split it for left or right view.

        Args:
            side (str, optional): Specify 'left' or 'right' to select the side of the image. Defaults to 'left'.

        Returns:
            np.ndarray: The left or right half of the frame from the video capture in BGR color space, or None if no frame is grabbed.
        """
        # Attempt to grab the next frame from the video capture
        ret, frame = self.vid.read()
        save_path = "/home/leonoor/MRIRAC_Leonoor/src/rcdt_LLM_fr3/Test13.jpg"

        # If a frame was successfully grabbed
        if ret:
            # Split the frame in half horizontally
            mid_point = frame.shape[1] // 2
            if side == "left":
                frame = frame[:, :mid_point]  # Get the left half
            elif side == "right":
                frame = frame[:, mid_point:]  # Get the right half

            else:
                raise ValueError("Invalid side specified. Choose 'left' or 'right'.")

            return frame
        else:
            rospy.logerr("Failed to grab frame")
            return None

    def detect_markers(self, frame, ARUCO_DICT, show=False, save_image=False):
        """Detect ArUco markers in the provided frame for each dictionary in ARUCO_DICT and return corners and ids."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        all_corners = []
        all_ids = []
        # Iterate over each ArUco dictionary type
        for aruco_dict_name, aruco_dict_id in ARUCO_DICT.items():
            aruco_dict = aruco.getPredefinedDictionary(aruco_dict_id)
            corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict, parameters=self.parameters)

            if ids is not None:
                all_corners.extend(corners)
                all_ids.extend(ids.flatten())  # Flatten the ids array
                aruco.drawDetectedMarkers(frame, corners, ids)  # Draw detected markers on the frame
                # Customize font for marker ID display
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 6.0  # Adjust scale for better visibility
                font_color = (255, 0, 0)  # Red color in BGR for better visibility
                font_thickness = 15  # Increase thickness for better visibility

                # Draw marker IDs at the top-left corner of the detected markers
                for i, corner in enumerate(corners):
                    x, y = int(corner[0][0][0]), int(corner[0][0][1])  # top-left corner
                    cv2.putText(frame, str(ids[i][0]), (x, y), font, font_scale, font_color, font_thickness)

                for i, marker_id in enumerate(ids.flatten()):  # Make sure ids are flattened here too
                    marker_id_str = str(marker_id)
                    # Initialize the dictionary entry for each detected marker
                    if marker_id_str not in self.detected_markers:
                        self.detected_markers[marker_id_str] = {}
                    self.detected_markers[marker_id_str]['corners2img'] = corners[i]
                    self.detected_markers[marker_id_str]['pos2img'] = np.mean(corners[i], axis=0, dtype=np.uint8)

        # Convert lists to numpy arrays for consistency with OpenCV functions
        corners = np.array(all_corners, dtype=np.float32) if len(all_corners) > 0 else None
        ids = np.array(all_ids) if len(all_ids) > 0 else None

        if save_image:
            save_path = "/home/allianderai/LLM-franka-control-rt/src/rcdt_LLM_fr3/MRIRAC_control/out/demo/detected_markers.jpg"
            cv2.imwrite(save_path, frame)

        if show:
            cv2.imshow("Detected Markers", frame)
            cv2.waitKey(1)

        return corners, ids

    def locate_markers(self, corners, ids, marker_size=0.04, refine=False):
        """locate marker coordinates to the sl.IMAGE coordinate frame by cv2.aruco"""
        #self._load_calibration(self.cur_side)

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, self.camera_matrix, self.dist_coeff
        )

        rvec, tvec = rvec.squeeze(), tvec.squeeze()
        #print(f"Calculated tvec for Marker ID {tvec}")

        for i, marker_id in enumerate(ids.flatten()):  # Ensure ids are flattened here
            marker_id_str = str(marker_id)
            if marker_id_str in self.detected_markers:
                self.detected_markers[marker_id_str]["pos2camera"] = tvec[i]
                self.detected_markers[marker_id_str]["rot2camera"] = rvec[i]
            else:
                rospy.logwarn(f"Warning: Marker {marker_id_str} not found in detected markers.")

        if refine:
            valid_res = 1
            for i in range(100):
                if valid_res > 9:
                    break
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_size, self.camera_matrix, self.dist_coeff
                )
                rvec, tvec = rvec.squeeze(), tvec.squeeze()
                valid_res += 1
                for i, id in enumerate(ids):
                    self.detected_markers[f"{id}"]["pos2camera"] += tvec[i]
            pos = [self.detected_markers[f"{id}"]["pos2camera"] for id in ids]
            for id in ids:
                self.detected_markers[f"{id}"]["pos2camera"] /= valid_res


if __name__ == "__main__":
    try:
        rospy.init_node('camera_node', anonymous=True)
        camera_index = rospy.get_param('~camera_index', 4)
        my_cam = Camera(camera_index, "CameraNode", marker_size=0.04)

        while not rospy.is_shutdown():
            frame = my_cam.get_img("right")
            if frame is not None:
                corners, ids = my_cam.detect_markers(frame, ARUCO_DICT, show=True, save_image=True)
                if ids is not None:
                    my_cam.locate_markers(corners, ids, my_cam.marker_size)
                    for marker_id in ids.flatten():
                        tvec = my_cam.detected_markers[str(marker_id)].get("pos2camera", None)
                        if tvec is not None:
                            rospy.loginfo(f"Marker ID: {marker_id}, tvec: {tvec}")
                        else:
                            rospy.loginfo(f"Marker ID: {marker_id} has no position vector (tvec).")

            cv2.waitKey(10)
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
        my_cam.close()
