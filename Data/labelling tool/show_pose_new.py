# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:32:58 2024

@author: Longfei
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import threading
import math



CONNECTIONS = [
    (0,14), (0,15),
    (0, 1), (1, 2),  
    (1, 5), (5, 6), 
    (6, 7),
    (2, 3), (3, 4),  
    (1, 8), (1, 11),  
    (8, 9), (9, 10), 
    (11, 12), (12, 13) 
]

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]


def add_bars_images(image, mv_scale, mv_speed, inact):
    font = cv2.FONT_HERSHEY_SIMPLEX  
    [m, n, d] = image.shape

    # Add speed bars
    plot = np.zeros((image.shape[0], 30, 3), dtype=np.uint8)
    height = int(mv_speed / 2 * plot.shape[0])
    cv2.rectangle(plot, (0, plot.shape[0] - height), (plot.shape[1], plot.shape[0]), (0, 255, 255), -1)
    
    # Place the label at the top of the bar
    text = 'vel'  # example text
    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_x = (plot.shape[1] - text_size[0]) // 2  # Horizontally center the text
    text_y = 20  # 20 pixels from the top
    cv2.putText(plot, text, (text_x, text_y), font, 0.5, (128, 0, 255), 1, cv2.LINE_AA)
    
    image1 = np.hstack((image, plot))

    # Add size bars
    plot = np.zeros((image1.shape[0], 30, 3), dtype=np.uint8)
    height = int(mv_scale / 1.5 * plot.shape[0])
    cv2.rectangle(plot, (0, plot.shape[0] - height), (plot.shape[1], plot.shape[0]), (255, 255, 0), -1)

    # Place the label at the top of the bar
    text = 'scl'  # example text
    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_x = (plot.shape[1] - text_size[0]) // 2
    text_y = 20
    cv2.putText(plot, text, (text_x, text_y), font, 0.5, (255, 0, 128), 1, cv2.LINE_AA)

    image1 = np.hstack((image1, plot))

    # Add inactivity bars
    plot = np.zeros((image1.shape[0], 30, 3), dtype=np.uint8)
    height = int(inact / 20 * plot.shape[0])
    cv2.rectangle(plot, (0, plot.shape[0] - height), (plot.shape[1], plot.shape[0]), (0, 0, 255), -1)

    # Place the label at the top of the bar
    text = 'ina'  # example text
    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_x = (plot.shape[1] - text_size[0]) // 2
    text_y = 20
    cv2.putText(plot, text, (text_x, text_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    image1 = np.hstack((image1, plot))
    return image1



def add_text_to_image(image, mv_scale, mv_speed, inact):
    # Define font and position for text (top-right corner)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White color
    thickness = 1

    # Prepare the text to be displayed
    text = f"Scale: {mv_scale:.2f}  Speed: {mv_speed:.2f}  Inactivity: {inact}s"

    # Get the size of the text to calculate the position
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Position to place the text in the top-right corner
    position = (image.shape[1] - text_width - 20, 40)  # 20 pixels margin from the right and top

    # Add the text to the image
    cv2.putText(image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    image = add_bars_images(image, mv_scale, mv_speed, inact)

    return image   



    
def filterout_pose(D2_pose, con_scores):
    D2_pose_filtered = []
    D2_score_filtered = []
    # if len(D2_pose)<=1:
    #     return D2_pose
    # else:
    for jj in range(0, len(D2_pose)):
        pose_coords = D2_pose[jj]
        score = con_scores[jj]
        if score>=5:
            D2_pose_filtered.append(pose_coords)
            D2_score_filtered.append(score)
    if len(D2_pose_filtered)>1:
        max_value = max(D2_score_filtered)
        max_indices = [i for i, x in enumerate(D2_score_filtered) if x == max_value]
        max_index = max_indices[0]
        D2_pose_filtered = [D2_pose_filtered[max_index]]
        # print('filtered: ', D2_pose_filtered)
    return D2_pose_filtered
    
   
    
   
def interpolate_joints(joints, window_size):
    # Get the shape of the data
    joints[joints == -1] = np.nan
    num_frames, num_joints, _ = joints.shape
    

    # Iterate through each joint and each axis (x, y)
    for joint_idx in range(num_joints):
        for axis in range(2):  # 0 for x, 1 for y
            # Extract the 1D array of positions for this joint and axis
            joint_data = joints[:, joint_idx, axis]
            # Create a pandas series to interpolate
            joint_series = pd.Series(joint_data)
            # Interpolate missing values (NaN)
            joint_series = joint_series.interpolate(method='linear', limit_direction='both', limit_area = 'inside')
            # Fill back to the array
            ttt = joint_series.values
            # ttt[np.isnan(ttt)] = -1
            joints[:, joint_idx, axis] = ttt
            
    # Apply a moving average to smooth joint trajectories
    smoothed_joints = np.copy(joints)
    for joint_idx in range(joints.shape[1]):  # Iterate through each joint
        for axis in range(2):  # Iterate through x and y
            ttt = pd.Series(joints[:, joint_idx, axis]).rolling(window=window_size, min_periods=1, center=True).mean()
            smoothed_joints[:, joint_idx, axis] = ttt
    return smoothed_joints



def smooth_all_poses_segment(data_seg):
    D2_pose_smooth = []
    mv_scale_smooth = []
    mv_speed_smooth = []
    inact_smooth = []
    obj_smooth = []
    for i_n in range(0, len(data_seg)):
        poses = data_seg[i_n]['poses']
        D2_pose = poses[2]
        con_scores = poses[1]
        mv_scale = data_seg[i_n]['mv_body_r']
        mv_speed = data_seg[i_n]['mv_sp']
        inact = data_seg[i_n]['nomv_sec']
        obj = data_seg[i_n]['objects']
        if len(D2_pose)>1:
            D2_pose = filterout_pose(D2_pose, con_scores)
        if len(D2_pose)<1:
            continue
        D2_pose_smooth.append(D2_pose[0])
        mv_scale_smooth.append(mv_scale)
        mv_speed_smooth.append(mv_speed)
        inact_smooth.append(inact)
        obj_smooth.append(obj)
    if len(D2_pose_smooth)<5:
        return D2_pose_smooth, mv_scale_smooth, mv_speed_smooth, inact_smooth, obj_smooth
    D2_pose_smooth = np.array(D2_pose_smooth).astype(float)
    D2_pose_smooth = interpolate_joints(D2_pose_smooth, window_size=3)
    return D2_pose_smooth, mv_scale_smooth, mv_speed_smooth, inact_smooth, obj_smooth
    

def plot_pose_this(pose_coords):
    # Load the image where you want to visualize the pose (use a blank image for testing)
    # image = np.zeros((480, 672, 3), dtype=np.uint8)  # Black background (1000x1000)
    pose_coords[np.isnan(pose_coords)] = -1
    image = np.zeros((600, 800, 3), dtype=np.uint8)  # Black background (1000x1000)
    for i, (x, y) in enumerate(pose_coords):
        # print('x:', x)
        # print('y:', y)
        if x != -1 and y != -1 :  # Check if the joint is not missing
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw joint as a green circle
    # Draw the lines between the joints
    for connection in CONNECTIONS:
        joint1, joint2 = connection
        x1, y1 = pose_coords[joint1]
        x2, y2 = pose_coords[joint2]
        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:  # Only draw if both joints are not missing
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[0], 2)  # Draw line in blue
    return image
     

def plot_pose(pose_coords):
    # Load the image where you want to visualize the pose (use a blank image for testing)
    # image = np.zeros((480, 672, 3), dtype=np.uint8)  # Black background (1000x1000)
    # pose_coords[np.isnan(pose_coords)] = -1
    image = np.zeros((600, 800, 3), dtype=np.uint8)  # Black background (1000x1000)
    for i, (x, y) in enumerate(pose_coords):
        # print('x:', x)
        # print('y:', y)
        if x != -1 and y != -1 :  # Check if the joint is not missing
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw joint as a green circle
    # Draw the lines between the joints
    for connection in CONNECTIONS:
        joint1, joint2 = connection
        x1, y1 = pose_coords[joint1]
        x2, y2 = pose_coords[joint2]
        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:  # Only draw if both joints are not missing
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[0], 2)  # Draw line in blue
    return image
     


def plot_objects(image, A):
    # Colors for the boxes and text
    box_colors = [(0, 255, 0), (255, 0, 0)]  # Green and Blue for boxes
    text_color = (0, 0, 0)  # Black for text
    
    # Draw bounding boxes and text
    for i in range(len(A[0])):
        bbox = A[2][i]
        confidence = A[1][i]
        label = f"{A[0][i]} ({confidence:.2f})"
    
        # Draw rectangle for the bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_colors[i % len(box_colors)], 2)
    
        # Put text label above the bounding box
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = bbox[0]
        text_y = max(bbox[1] - 5, text_size[1] + 5)
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (255, 255, 255), -1)
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return image
    

def show_data_pose(data_seg, stop_event):
    D2_pose_smooth, mv_scale_smooth, mv_speed_smooth, inact_smooth, obj_smooth = smooth_all_poses_segment(data_seg)
    if len(D2_pose_smooth)<10:
        return
    

    
    for jj in range(0, len(D2_pose_smooth)):
        if stop_event is not None and stop_event.is_set():  # If stop event is triggered, break out of the loop
            break
        
        pose_coords = D2_pose_smooth[jj]
        
        #     print(pose_coords)
        image = plot_pose_this(pose_coords)
    
        image = add_text_to_image(image, mv_scale_smooth[jj], mv_speed_smooth[jj], inact_smooth[jj])
        
        # image = plot_objects(image, obj_smooth[jj])
        
        cv2.imshow("Video", image)
        if cv2.waitKey(100) & 0xFF == ord('n'):  # Close on 'q' key
            break
        # # Save frame as an image file
        # frame_path = os.path.join(f"frame_{jj:03d}.png")
        # cv2.imwrite(frame_path, image)
    # Close all OpenCV windows
    # cv2.waitKey(500)
    cv2.destroyAllWindows()
    

    

    
if __name__ == "__main__":
    pkl_file = 'G:\\home visit\\users012\\20240822.pkl'
    stop_event = threading.Event()
    stop_event.clear()
    if pkl_file:
        with open(pkl_file, 'rb') as file:
            pkl_file_name = os.path.basename(pkl_file)
            print(f"Loaded .pkl file: {pkl_file_name}")
    
            video_file = pickle.load(file)
            for h in range(17, len(video_file)):
                for m in range(5,len(video_file[0])):
                    for s in range(3, len(video_file[0][0])):
                        data_seg = video_file[h][m][s]
                        if len(data_seg) <= 20:
                            continue
                        show_data_pose(data_seg, stop_event)