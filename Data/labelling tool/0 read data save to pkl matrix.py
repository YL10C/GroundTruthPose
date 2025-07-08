import os
import json
import numpy as np
import datetime
import pickle
import pytz
import cv2



def load_video_to_memory(org_video_path):
    # Open the video file
    cap = cv2.VideoCapture(org_video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file {org_video_path}")
    
    frames = []
    
    # Read frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop reading when no more frames are available
        
        frames.append(frame)  # Append each frame to the list
    
    # Release the video capture object
    cap.release()
    
    return frames


def time_covert(timestamp_ms):
    # Convert from milliseconds to seconds
    timestamp_s = timestamp_ms / 1000
    # Convert to a human-readable format
    dt = datetime.datetime.fromtimestamp(timestamp_s, datetime.UTC)
    london_tz = pytz.timezone('Europe/London')
    dt_london = dt.astimezone(london_tz)
    dt = dt_london
    # Print the formatted date and time
    # Extract individual components
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second
    milliseconds = timestamp_ms % 1000
    return [year, month, day, hour, minute, second, milliseconds]


# Define the main folder path
user_id = 'p1'
events = 'user_inter_001_living_mindful_meal'

user_id = 'p2'
events = 'user_inter_002_living_art'

main_folder = f'G:\\intervention data\\{user_id}\\{events}'


subfolders = os.listdir(main_folder)
subfolders.sort()

for subfolder in subfolders:
    subfolder_path = os.path.join(main_folder, subfolder)
    

    if os.path.isfile(subfolder_path + '.pkl'):
        continue
    if  '20250314' in subfolder_path  or '20250323' in subfolder_path:
        continue
    
    # Check if the path is indeed a directory (subfolder)
    if os.path.isdir(subfolder_path):
        data_hour_min = [[[[] for _ in range(6)] for _ in range(60)] for _ in range(24)]
        
        
        # Go through files in each subfolder
        for file in os.listdir(subfolder_path):
            if file.endswith('data.json'):
                json_path = os.path.join(subfolder_path, file)
                try:
                    with open(json_path, 'r') as f:
                        print(file)
                        data = json.load(f)
                        print('data len: ', len(data))
                        # org_video_path = file.split('_')[0] + '_' + file.split('_')[1] + '_' +  file.split('_')[2] +'_vid.mp4' 
                        # org_video_path = os.path.join(subfolder_path, org_video_path)
                        # frames = load_video_to_memory(org_video_path)
                        # print('frame len: ', len(frames))
                        # Store data with path as key for easy identification
                        for line in data:
                            time_stamp_line = line['t_stamp']
                            time_line =  time_covert(time_stamp_line)
                            # print(time_line)
                            hour_line = time_line[3]
                            minute_line = time_line[4]
                            second_line = int(time_line[5]/10)
                            data_hour_min[hour_line][minute_line][second_line].append(line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {json_path}")
                            

        # Save the list to a JSON file
        file_path = main_folder + r'.\{0}{1:02}{2:02}.pkl'.format(time_line[0], time_line[1], time_line[2])
            
        with open(file_path, 'wb') as f:
            pickle.dump(data_hour_min, f)




