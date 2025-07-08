# 1 play_video_test_newnew_version_fixed.py
# 完整版本，修复线程播放问题，使用主线程播放视频

import os
import tkinter as tk
from tkinter import filedialog
import pickle
import numpy as np
import csv
import pandas as pd
from collections import Counter
from show_pose_new import show_data_pose

def save_data_to_csv(all_data_labelled, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filepath', 'hour', 'minute', 'second', 'label'])  # 写入表头
        for row in all_data_labelled:
            writer.writerow(row)

def label_video(all_data_labelled, label, info):
    date = info[0]
    info_new = [date, info[1], info[2], info[3], label]
    all_data_labelled.append(info_new)
    return all_data_labelled

def create_label_window(on_label_button_click):
    root = tk.Tk()
    root.title("Video Labeling Tool")
    root.geometry("350x540")
    label_var = tk.StringVar()
    label_var.set("Select the label for the current segment:")

    tk.Label(root, textvariable=label_var, font=("Helvetica", 14)).pack(pady=10)

    labels = {
        "Dow no [1]": 1,
        "Dow med [2]": 2,
        "Dow hig [3]": 3,
        "Sta no [4]": 4,
        "Sta med [5]": 5,
        "Std hig [6]": 6,
        "Oth [7]": 7,
        "Nex [N]": 0,
        "Replay (A)": -1,
        'Forward m+1': 111,
        'Forward m+2': 222,
        'Forward m+5': 555,
        'Save (S)': 999,
    }

    def on_key_press(event):
        key_map = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, 'n': 0, 'a': -1, 's': 999}
        if event.char.lower() in key_map:
            on_label_button_click(key_map[event.char.lower()])

    for label_text, label_value in labels.items():
        tk.Button(root, text=label_text, command=lambda val=label_value: on_label_button_click(val)).pack(pady=5)

    root.deiconify()
    root.focus_force()
    root.attributes('-topmost', 1)
    root.bind('<KeyPress>', on_key_press)
    return root, label_var

def print_label_occurance(all_data_labelled):
    last_elements = [sublist[-1] for sublist in all_data_labelled]
    counts = Counter(last_elements)
    labels = ['sit0', 'sit1', 'sit2', 'std0', 'std1', 'std2', 'oth']
    for number in range(1, 8):
        print(f"{number}.{labels[number-1]}: {counts.get(number, 0)}")

# === 脚本入口 ===
filename = './label3.csv'
if os.path.exists(filename) and os.path.getsize(filename) > 0:
    df = pd.read_csv(filename)
    all_data_labelled = df.values.tolist()
else:
    all_data_labelled = []

pkl_file = filedialog.askopenfilename(
    title="Select a .pkl File",
    filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*"))
)

if pkl_file:
    with open(pkl_file, 'rb') as file:
        pkl_file_name = os.path.basename(pkl_file)
        date = pkl_file_name.split('.')[0]
        url_full = os.path.abspath(pkl_file)
        print(f"Loaded .pkl file: {pkl_file_name}")

        video_file = pickle.load(file)
        label_selected = None
        hour = 18

        root, label_var = create_label_window(lambda val: globals().__setitem__('label_selected', val))

        for h in range(hour, hour + 1):
            print("Checking hour:", h)
            data_hour = video_file[h]
            data_hour_flaten = [item for sublist in data_hour for item in sublist]

            if np.sum([len(sublist) for sublist in data_hour_flaten]) < 300:
                continue

            m = 0
            while m < 60:
                s = 0
                while s < 6:
                    data_seg = video_file[h][m][s]
                    if len(data_seg) <= 15:
                        s += 1
                        continue

                    print('%s %s:%s %s' % (date, h, m, s * 10))
                    show_data_pose(data_seg, stop_event=None)  # 主线程播放

                    label_selected = None
                    while label_selected is None or label_selected == -1:
                        root.update()

                    if label_selected in range(1, 8):
                        info = [url_full, h, m, s * 10]
                        all_data_labelled = label_video(all_data_labelled, label_selected, info)
                        print_label_occurance(all_data_labelled)
                        s += 1

                    elif label_selected == 0:
                        s += 1
                    elif label_selected == -1:
                        pass
                    elif label_selected == 111:
                        m = min(m + 1, 59); s = 0
                    elif label_selected == 222:
                        m = min(m + 2, 59); s = 0
                    elif label_selected == 555:
                        m = min(m + 5, 59); s = 0
                    elif label_selected == 999:
                        save_data_to_csv(all_data_labelled, filename)
                    else:
                        s += 1
                m += 1

        save_data_to_csv(all_data_labelled, filename)
        root.destroy()
