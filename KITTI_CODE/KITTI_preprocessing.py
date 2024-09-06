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
        cv_data_list (list): FrameID ObjectID t1 t2 t3 r1 in cv format

    Returns:
        list: SE(3) homogeneous matrix 4x4
    """
    # FrameID ObjectID t1 t2 t3 r1
    # Where ti are the coefficients of 3D object location **t** in camera coordinates, and r1 is the Rotation around
    # Y-axis in camera coordinates.
    assert(len(cv_data_list) == 6)

    # Extract the translation vector
    t = np.array([
        cv_data_list[2],
        cv_data_list[3],
        cv_data_list[4]
    ], dtype=np.float64);

    # Extract rotation angles and convert to radians
    y = float(cv_data_list[5]) + (np.pi / 2)
    x = 0.0
    z = 0.0

    # Compute the rotation matrix elements
    cy = np.cos(y)
    sy = np.sin(y)
    cx = np.cos(x)
    sx = np.sin(x)
    cz = np.cos(z)
    sz = np.sin(z)

    m00 = cy * cz + sy * sx * sz
    m01 = -cy * sz + sy * sx * cz
    m02 = sy * cx
    m10 = cx * sz
    m11 = cx * cz
    m12 = -sx
    m20 = -sy * cz + cy * sx * sz
    m21 = sy * sz + cy * sx * cz
    m22 = cy * cx

    # Construct the rotation matrix
    R = np.array([
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22]
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
    """ Take a list in cv format and return a normal list in: FrameID ObjectID x y z yaw(z) 

    Args:
        cv_data_list (np.array): list in cv format: FrameID ObjectID t1 t2 t3 r1
    """
    ingredient_1 = camera_coordinate_to_world()
    ingredient_2 = constructObjectPoseGT(cv_data_list)
    homogeneous_matrix = ingredient_1 @ ingredient_2 @ np.linalg.inv(ingredient_1)
    
    translation_vector = homogeneous_matrix[0:3, 3]
    rotation_matrix = homogeneous_matrix[0:3, 0:3]
    
    rotation = R.from_dcm(rotation_matrix)
    euler_angles = rotation.as_euler('zyx', degrees=False)  # Yaw, Pitch, Roll

    # Extract individual angles
    yaw, _, _ = euler_angles
    to_return = np.concatenate([cv_data_list[0:2], translation_vector, np.array([yaw])])
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




base_path = 'KITTI_data'
all_folders = os.listdir('KITTI_data') # Directory with all the data
# TODO: Assert is directory



def process_data(input_data):
    maybe_makedirs('../KITTI_CODE/KITTI_data_processed') # output data
    
    columns_to_extract = [1, 2, 7, 8, 9, 10]
    column_names_cv = ['Scene_id', 'Frame_id', 'Object_id', 'cv_x', 'cv_y', 'cv_z', 'cv_yaw']
    column_names = ['Scene_id', 'Frame_id', 'Object_id', 'x', 'y', 'z', 'yaw']
    data = []
    data_cv = []
    
    for folder in all_folders:
        folder_name = folder
        txt_file_path = os.path.join('KITTI_data', folder_name, 'object_pose.txt')
        
        if os.path.isfile(txt_file_path):
            # Open and read the .txt file
            with open(txt_file_path, 'r') as file:
                for line in file:
                    # Split the line into values
                    values = line.split()
                    # Extract the specified columns and add them to the data list
                    extracted_values = [values[i - 1] for i in columns_to_extract]
                    data_cv.append([folder_name] + extracted_values)
                    normal_values = cv_to_normal(extracted_values)
                    data.append([folder_name] + normal_values)
                
        else:
            print(f"File {txt_file_path} does not exist. x1")
            #sys.exit(f"File {txt_file_path} does not exist. Processing terminated.")
    
    df = pd.DataFrame(data_cv, columns=column_names_cv)
    df2 = pd.DataFrame(data, columns=column_names)
    
    csv_file_path_cv = os.path.join('KITTI_data_processed', 'kitti_processed_data_cv.csv')
    csv_file_path_normal = os.path.join('KITTI_data_processed', 'kitti_processed_data_normal.csv')
    df.to_csv(csv_file_path_cv, index=False)
    df2.to_csv(csv_file_path_normal, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=False)
    args = parser.parse_args()
    process_data(args.data)

