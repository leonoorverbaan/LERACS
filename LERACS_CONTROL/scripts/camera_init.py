import os
import cv2
import cv2.aruco as aruco
import rospy
import numpy as np
import time

# Define names of ArUco tags supported by OpenCV
ARUCO_DICT = {
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
}

class Camera:
    def __init__(self, camera_index=4, name="default", marker_size=0.04):
        """
        Initialize the Camera object.

        Args:
            camera_index (int): Index of the camera to use.
            name (str): Name of the camera.
            marker_size (float): Size of the ArUco markers.
        """
        self.name = name
        self.video_source = camera_index
        self.parameters = aruco.DetectorParameters()
        self.marker_size = marker_size
        self.cur_side = "right"  # Record the last call for the side of get_image

        # Camera intrinsics (default is the left camera)
        self.camera_matrix_left = np.array([[1069.86, 0.0, 929.96],
                                            [0.0, 1069.8101, 540.947],
                                            [0.0, 0.0, 1.0]])
        self.dist_coeff_left = np.array([-0.0458, 0.0162, -0.0001, -0.0009, 0.0068])
        self.camera_matrix_right = np.array([[387.0155334472656, 0, 316.3547668457031],
                                             [0, 387.0155334472656, 243.0697021484375],
                                             [0, 0, 1]])
        self.dist_coeff_right = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        self._load_calibration(self.cur_side)
        self.marker_pixel_coordinates = None
        self.detected_markers = dict()

        self._init_streaming()

    def _init_streaming(self):
        """
        Initialize video capture with the camera index.
        """
        # Open video capture with the specified video source
        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_V4L2)

        # Check if the video capture opened successfully
        if not self.vid.isOpened():
            rospy.logerr(f"Unable to open video source: {self.video_source}")
            raise IOError(f"Failed to open video source: {self.video_source}")

        # Set video capture properties
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.vid.set(cv2.CAP_PROP_FOURCC, fourcc)

        rospy.loginfo(f"Camera {self.name} streaming from source {self.video_source}")

    def _load_calibration(self, side="right"):
        """
        Load camera calibration for the specified side.

        Args:
            side (str): Specify 'left' or 'right' to select the side of the calibration.
        """
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
        """
        Release the video capture object.
        """
        self.vid.release()

    def reset(self):
        """
        Reset the video capture.
        """
        self.close()
        time.sleep(0.3)
        self._init_streaming()
        time.sleep(1)

    def get_img(self, side: str = "right") -> np.ndarray:
        """
        Grab a frame from the video capture and split it for left or right view.

        Args:
            side (str, optional): Specify 'left' or 'right' to select the side of the image. Defaults to 'right'.

        Returns:
            np.ndarray: The left or right half of the frame from the video capture in BGR color space, or None if no frame is grabbed.
        """
        # Attempt to grab the next frame from the video capture
        ret, frame = self.vid.read()

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
        """
        Detect ArUco markers in the provided frame for each dictionary in ARUCO_DICT and return corners and ids.

        Args:
            frame (np.ndarray): The image frame to detect markers in.
            ARUCO_DICT (dict): Dictionary of ArUco marker types to detect.
            show (bool, optional): Whether to display the image with detected markers. Defaults to False.
            save_image (bool, optional): Whether to save the image with detected markers. Defaults to False.

        Returns:
            tuple: Tuple of corners and ids of detected markers.
        """
        # Convert the frame to grayscale
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

                # Store detected marker data
                for i, marker_id in enumerate(ids.flatten()):
                    marker_id_str = str(marker_id)
                    if marker_id_str not in self.detected_markers:
                        self.detected_markers[marker_id_str] = {}
                    self.detected_markers[marker_id_str]['corners2img'] = corners[i]
                    self.detected_markers[marker_id_str]['pos2img'] = np.mean(corners[i], axis=0, dtype=np.uint8)

        # Convert lists to numpy arrays for consistency with OpenCV functions
        corners = np.array(all_corners, dtype=np.float32) if len(all_corners) > 0 else None
        ids = np.array(all_ids) if len(all_ids) > 0 else None

        # Save the image with detected markers if requested
        if save_image:
            save_path = os.path.join('..', 'out', 'demo', 'detected_markers.jpg')
            cv2.imwrite(save_path, frame)

        # Display the image with detected markers if requested
        if show:
            cv2.imshow("Detected Markers", frame)
            cv2.waitKey(1)

        return corners, ids

    def locate_markers(self, corners, ids, marker_size=0.04, refine=False):
        """
        Locate marker coordinates to the sl.IMAGE coordinate frame by cv2.aruco.

        Args:
            corners (np.ndarray): Corners of the detected markers.
            ids (np.ndarray): IDs of the detected markers.
            marker_size (float, optional): Size of the markers. Defaults to 0.04.
            refine (bool, optional): Whether to refine the marker locations. Defaults to False.
        """
        # Estimate the pose of the markers
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, self.camera_matrix, self.dist_coeff
        )

        rvec, tvec = rvec.squeeze(), tvec.squeeze()

        # Store the position and rotation vectors of detected markers
        for i, marker_id in enumerate(ids.flatten()):
            marker_id_str = str(marker_id)
            if marker_id_str in self.detected_markers:
                self.detected_markers[marker_id_str]["pos2camera"] = tvec[i]
                self.detected_markers[marker_id_str]["rot2camera"] = rvec[i]
            else:
                rospy.logwarn(f"Warning: Marker {marker_id_str} not found in detected markers.")

        # Refine the marker locations if requested
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
        # Initialize the ROS node
        rospy.init_node('camera_node', anonymous=True)

        # Get the camera index parameter
        camera_index = rospy.get_param('~camera_index', 4)

        # Initialize the Camera object
        my_cam = Camera(camera_index, "CameraNode", marker_size=0.04)

        # Main loop
        while not rospy.is_shutdown():
            # Get the image from the right camera
            frame = my_cam.get_img("right")
            if frame is not None:
                # Detect markers in the image
                corners, ids = my_cam.detect_markers(frame, ARUCO_DICT, show=True, save_image=True)
                if ids is not None:
                    # Locate markers in the image
                    my_cam.locate_markers(corners, ids, my_cam.marker_size)
                    for marker_id in ids.flatten():
                        tvec = my_cam.detected_markers[str(marker_id)].get("pos2camera", None)
                        if tvec is not None:
                            rospy.loginfo(f"Marker ID: {marker_id}, tvec: {tvec}")
                        else:
                            rospy.loginfo(f"Marker ID: {marker_id} has no position vector (tvec).")

            # Wait briefly to avoid excessive CPU usage
            cv2.waitKey(10)
    except rospy.ROSInterruptException:
        pass
    finally:
        # Clean up
        cv2.destroyAllWindows()
        my_cam.close()
