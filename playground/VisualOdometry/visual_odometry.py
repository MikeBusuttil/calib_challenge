import os
import numpy as np
import pandas as pd
import cv2
import math

from lib.visualization import plotting
from lib.visualization.video import play_trip
from lib.visualization.colorize import colorize

from tqdm import tqdm
from sh import Command
from sigfig import round


class VisualOdometry():
    """
    
    Attributes
    ----------
    K (ndarray): Intrinsic matrix
    P (ndarray): Projection matrix
    """
    def __init__(self, src):
        if ".hevc" in src:
            self.images = self._load_videos(src)
            self.K, self.P = self._guess_calibration(self.images[0])
            self.gt_poses = self._load_poses(os.path.join("KITTI_sequence_2", "poses.txt"))
        else:
            self.K, self.P = self._load_calib(os.path.join(src, 'calib.txt'))
            self.gt_poses = self._load_poses(os.path.join(src, "poses.txt"))
            self.images = self._load_images(os.path.join(src,"image_l"))

        MAX_FEATURES = 3_000
        self.orb = cv2.ORB_create(MAX_FEATURES)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _guess_calibration(image):
        """
        Guesses the calibration of the camera

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        f = 910
        height, width, _ = image.shape
        K = np.ndarray([
            [ f/width, 0,        width/2  ],
            [ 0,       f/height, height/2 ],
            [ 0,       0,        1        ]
        ])
        # Note: the center of the sensor differs a bit from the optical center of the lens system
        # which means the c_x (width/2) & c_y (height/2) values are off by a little bit
        P = np.ndarray(4,4)
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_videos(src):
        """
        Loads the video

        Parameters
        ----------
        src (str): The file path to video

        Returns
        -------
        images (list): grayscale images
        """
        cap = cv2.VideoCapture(src)
        images = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('cant read?')
                break
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return images

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    @staticmethod
    def points_distance(p0, p1):
        return math.sqrt((p0[0]-p1[0])**2+(p0[1]-p1[1])**2)
    
    @staticmethod
    def points_relative(p0, p1):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return (p0[0] + dx, p0[1] + dy)

    def get_matches(self, f):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        f (int): The current frame

        Returns
        -------
        q1 (ndarray(2-dim ndarray)): The good keypoints matches position in i-1'th image
        q2 (ndarray(2-dim ndarray)): The good keypoints matches position in i'th image
        """
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[f - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[f], None)

        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Filter out the matches with overly dissimilar "distance"'s.  For a description of "distance" see:
        # https://stackoverflow.com/questions/16996800/what-does-the-distance-attribute-in-dmatches-mean 
        _matches = []
        try:
            for m1, m2 in matches:
                if m1.distance < 0.8 * m2.distance:
                    _matches.append([m1,m2])
        except ValueError:
            pass
        print(f'{len(matches)} -> {len(_matches)}')
        matches = _matches

        #Filter out matches that indicate significant vertical movement (dirty hack)
        height_filter = False
        # height_filter = True
        if height_filter:
            rises = pd.DataFrame([kp2[m2.trainIdx].pt[1] - kp1[m1.queryIdx].pt[1] for m1, m2 in matches])
            _matches = []
            for m1, m2 in matches:
                if abs(kp2[m2.trainIdx].pt[1] - kp1[m1.queryIdx].pt[1]) < 2*rises.std()[0]:
                    _matches.append([m1,m2])
            print(f'{len(matches)} -> {len(_matches)}')
            matches = _matches

        #Filter out matches that indicate significant movement (doesn't always work in general)
        distance_filter = False
        distance_filter = True
        if distance_filter:
            distances = pd.DataFrame([self.points_distance(kp1[m1.queryIdx].pt, kp2[m2.trainIdx].pt) for m1, m2 in matches])
            # plotting.histogram(distances)
            max_distance = distances.std()[0]/2
            _matches = []
            for m1, m2 in matches:
                if self.points_distance(kp1[m1.queryIdx].pt, kp2[m2.trainIdx].pt) < max_distance:
                    _matches.append([m1,m2])
            print(f'{len(matches)} -> {len(_matches)}')
            matches = _matches

        #TODO: filter out points that differ wildly in magnitude or direction from their neighbors
        #TODO: filter out points on cars
        #TODO: look at some of the matches that got filtered out to make sure baby not being thrown out with bathwater
        #TODO: compare (numerically) the different filter strategy combinations in 1 run
        good = [m1 for m1, m2 in matches]

        custom_view = False
        custom_view = True
        if custom_view:
            imgL = cv2.cvtColor(np.ndarray.copy(self.images[f-1]), cv2.COLOR_GRAY2RGB)
            imgR = cv2.cvtColor(np.ndarray.copy(self.images[f]), cv2.COLOR_GRAY2RGB)
            for m, (m1, m2) in enumerate(matches):
                color = colorize(m*100/(len(matches) - 1))
                pL, pR = tuple(map(int, kp1[m1.queryIdx].pt)), tuple(map(int, kp2[m2.trainIdx].pt))
                imgL = cv2.circle(imgL, pL, 5, color)
                imgR = cv2.circle(imgR, pR, 5, color)
                imgL = cv2.line(imgL, pL, self.points_relative(pL, pR), color)
                imgR = cv2.line(imgR, pR, self.points_relative(pR, pL), color)
            img3 = np.concatenate((imgL, imgR), axis=0)
        else:
            draw_params = dict(
                matchesMask = None, # draw only inliers
                flags = 2
            )
            img3 = cv2.drawMatches(self.images[f-1], kp1, self.images[f], kp2, good, None, **draw_params)
        cv2.imshow("image", img3)
        pressed_key = cv2.waitKey(1)
        if pressed_key != -1:
            if pressed_key == ord('q'):
                exit()
            if cv2.waitKey() == ord('q'):
                exit()

        # Get the image points form the good matches
        # Note: kp2 uses trainIdx while kp1 uses queryIdx
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

    @staticmethod
    def get_euler_angles(T, units='radians'):
        """
        Calculates the Euler angles from a given transformation matrix using RzRyRx method

        where Ry is yaw, Rx is pitch, Rz is roll

        Parameters
        ----------
        T (ndarray):    Transformation Matrix
        units (string): Either 'degrees' or 'radians' (default)

        Returns
        -------
        roll (float):  the magnitude of the roll
        pitch (float): the magnitude of the pitch
        yaw (float):   the magnitude of the yaw
        """
        R = T[:3, :3]

        #TODO: ensure roll & pitch aren't interchanged
        #TODO: test 11 other orderings against pre-trained sets to see if the rotation order matters
        cosine_for_yaw = math.sqrt(R[0][0] ** 2 + R[1][0] ** 2)
        is_singular = cosine_for_yaw < 10**-6
        if not is_singular:
            pitch = math.atan2(R[1][0], R[0][0])
            yaw = math.atan2(-R[2][0], cosine_for_yaw)
            roll = math.atan2(R[2][1], R[2][2])
        else:
            print("no roll\n\n\n")
            pitch = math.atan2(-R[1][2], R[1][1])
            yaw = math.atan2(-R[2][0], cosine_for_yaw)
            roll = 0

        if units == 'degrees':
            return roll* 180 / math.pi, pitch * 180 / math.pi, yaw * 180 / math.pi
        return roll, pitch, yaw
    
    @staticmethod
    def present(num):
        sign = '-' if num < 0 else '+'
        return sign + round(abs(num), decimals=2, output=str).rjust(5)

def main():
    src = "../labeled/3.hevc"
    src = "KITTI_sequence_2"  # Try KITTI_sequence_2 too
    vo = VisualOdometry(src)

    # play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    roll_total, pitch_total, yaw_total = 0, 0, 0
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            roll, pitch, yaw = vo.get_euler_angles(transf, units='degrees')
            roll_total, pitch_total, yaw_total = roll + roll_total, pitch + pitch_total, yaw + yaw_total
            print(f'roll is {vo.present(roll)      }°, pitch is {vo.present(pitch)      }°, yaw is {vo.present(yaw)      }°')
            print(f'        {vo.present(roll_total)}°           {vo.present(pitch_total)}°         {vo.present(yaw_total)}°')
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    file_out=os.path.basename(src) + ".html"
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=file_out)
    Command('explorer.exe')(file_out, _ok_code=[0,1])


if __name__ == "__main__":
    main()
