"""
Processing KITTI object positions into x,y,z, roll, pitch yaw.
The output csv file is in a format:
Scene_id, Frame_id, category (vehicle, other), node_id (object_id), robot (ego = True/False), x, y, z, length, width, height, heading, orientation (quaternion for ego, Nnoe otherwise).
"""
    
import sys
import os
import numpy as np
import pandas as pd
import argparse
from scipy.spatial.transform import Rotation as R

def constructObjectPoseGT(cv_data_list):
    """
    Args:
        cv_data_list (list): FrameID t1 t2 t3 R11 R12 R13 R21 R22 R23 R31 R32 R33

    Returns:
        list: SE(3) homogeneous matrix 4x4
    """
    # FrameID t1 t2 t3 R11 R12 R13 R21 R22 R23 R31 R32 R33
    # Where ti are the coefficients of 3D object location **t** in camera coordinates, and R.. is a rotation matrix
    assert(len(cv_data_list) == 13)

    # Extract the translation vector
    t = np.array([
        cv_data_list[1],
        cv_data_list[2],
        cv_data_list[3]
    ], dtype=np.float64)

    # Construct the rotation matrix
    R = np.array([
        cv_data_list[4:7],
        cv_data_list[7:10],
        cv_data_list[10:13]
    ], dtype=np.float64)

    Pose = np.eye(4, dtype=np.float64)
    Pose[0:3, 0:3] = R
    Pose[0:3, 3] = t
    return Pose

def camera_coordinate_to_world() -> np.ndarray:
    """
    Returns:
        np.ndarray: 4x4 matrix for the transformation
    """
    translation_vector = np.array([0.0, 0.0, 0.0])
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]])
    transformation_matrix[0:3, 3] = translation_vector
    return(transformation_matrix)

def cv_to_normal(cv_data_list):
    """ Take a list in cv format and return a normal list in: FrameID t1 t2 t3 R11 R12 R13 R21 R22 R23 R31 R32 R33

    Args:
        cv_data_list (np.array): list in cv format: FrameID t1 t2 t3 roll pitch yaw
    """
    ingredient_1 = camera_coordinate_to_world()
    ingredient_2 = constructObjectPoseGT(cv_data_list)
    homogeneous_matrix = ingredient_1 @ ingredient_2 @ np.linalg.inv(ingredient_1)
    
    translation_vector = homogeneous_matrix[0:3, 3]
    rotation_matrix = homogeneous_matrix[0:3, 0:3]
    
    rotation = R.from_dcm(rotation_matrix)
    euler_angles = rotation.as_euler('zyx', degrees=False)  # Yaw, Pitch, Roll

    # Extract individual angles
    yaw, pitch, roll = euler_angles
    to_return = np.concatenate([[cv_data_list[0]], translation_vector, np.array([roll, pitch, yaw])])
    return to_return.tolist()
        
def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise




base_path = 'KITTI_data/data_for_train'
all_folders = os.listdir(base_path) # Directory with all the data
# TODO: Assert is directory



def process_data(input_data):
    maybe_makedirs('../KITTI_CODE/KITTI_data_processed') # output data
    
    column_names = ['Scene_id', 'Object_id', 'Frame_id', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
    data = []
    data_cv = []
    
    for folder in all_folders:
        folder_name = folder
        txt_file_path = os.path.join(base_path, folder_name, 'pose_gt.txt')
        
        if os.path.isfile(txt_file_path):
            # Open and read the .txt file
            with open(txt_file_path, 'r') as file:
                for line in file:
                    # Split the line into values
                    values = line.split()
                    # Extract the specified columns and add them to the data list
                    rotation = np.array([values[1:4], values[5:8], values[9:12]])
                    translation = np.array([values[4], values[8], values[12]])
                    
                    rotation_matrix = R.from_dcm(rotation)
                    data_cv = [values[0], values[4], values[8], values[12], values[1], values[2], values[3], values[5], values[6], values[7], values[9], values[10], values[11]]
                    
                    normal_values = cv_to_normal(data_cv)
                    
                    data.append([folder_name, 'ego'] + normal_values)
                
        else:
            print(f"File {txt_file_path} does not exist. x1")
            #sys.exit(f"File {txt_file_path} does not exist. Processing terminated.")

    df = pd.DataFrame(data, columns=column_names)
    
    csv_file_path = os.path.join('KITTI_data_processed', 'kitti_processed_data_camera_global.csv')
    df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=False)
    args = parser.parse_args()
    process_data(args.data)

