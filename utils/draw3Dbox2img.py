import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from lib.datasets.kitti.kitti_utils import Calibration, draw_projected_box3d
import cv2

       
def compute_3d_box_cam(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def read_detection(path, is_gt=False):

    # Check if file is empty
    if os.path.getsize(path) <= 0:
        return pd.DataFrame()
    df = pd.read_csv(path, header=None, sep=' ')
    if is_gt:
        df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                      'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']
    else:
        df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                      'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y', 'score']
#     df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
#     df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    df = df[df['type']=='Car']
    df.reset_index(drop=True, inplace=True)
    return df


def get_data(curr_path, img_id):
    path_img = os.path.join(curr_path, '../data/image_2/%06d.png'%img_id)

    calib = Calibration(os.path.join(curr_path, '../data/calib/%06d.txt'%img_id))
    ground_truth = read_detection(os.path.join(curr_path, '../data/label_2/%06d.txt'%img_id), is_gt=True)
    predicted = read_detection(os.path.join(curr_path, '../outputs/monodetr/%06d.txt'%img_id), is_gt=False)
    return calib, path_img, ground_truth, predicted




def plot_2d(path_img, df, gt, calib):
    """
    

    """
    image = cv2.imread(path_img)
    # Plot the ground truth 3D bounding boxes on the image
    for o in range(len(gt)):
        corners_3d = compute_3d_box_cam(*gt.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        pts_2d = calib.project_rect_to_image(corners_3d.T)
        image_gt = draw_projected_box3d(image, pts_2d, color=(0,255,0), thickness=1)

    # Plot the projected 3D bounding boxes on the image
    for o in range(len(df)):
        corners_3d = compute_3d_box_cam(*df.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        pts_2d = calib.project_rect_to_image(corners_3d.T)
        image_dt = draw_projected_box3d(image_gt, pts_2d, color=(255,0,255), thickness=1)

    img_id = path_img.split('/')[-1].split('.')[0]
    cv2.imwrite(str(img_id) + '_gt_and_pred.png', image_dt)




def draw_3d_box(ax, corners_3d_cam, color='b'):
    """
    Plot a 3D bounding box on a given Axes3D (ax)

    corners_3d_cam: 8 corners of the box in camera coordinates (3x8 array)
    color: color of the box
    """
    # Create connections between corners (indices in the corners array)
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connections between upper and lower planes
    ]

    for connection in connections:
        ax.plot(*corners_3d_cam[:, connection], c=color)

def plot_3d(df=None, gt=None):
    # Create a new 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot predicted 3D boxes
    if df is not None:
        for o in range(len(df)):
            corners_3d_df = compute_3d_box_cam(*df.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
            draw_3d_box(ax, corners_3d_df, color='purple')
    else:
        print('No predicted boxes to plot')

    # Plot ground truth 3D boxes
    if gt is not None:
        for o in range(len(gt)):
            corners_3d_gt = compute_3d_box_cam(*gt.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
            draw_3d_box(ax, corners_3d_gt, color='green')
    else:
        print('No ground truth boxes to plot')

    # Set plot limits and labels
    ax.set_xlim([-15, 15])
    ax.set_ylim([0, 30])
    ax.set_zlim([0, 30])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the figure
    plt.show()


if __name__ == '__main__':
    img_id = 8
    cur_path = os.path.dirname(os.path.realpath(__file__))


    calib, path_img, gt, df = get_data(cur_path, img_id)

    plot_2d(path_img, df, gt, calib)

    plot_3d(df, gt)