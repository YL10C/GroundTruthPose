import pandas as pd

# window_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 51, 101]
# file_name = 'velocity_window_'
file_name = 'velocity_multiwindow_mean'
window_sizes = [[7, 9, 11], [7, 9, 11, 13], [7, 9, 11, 13, 15], [5, 7, 9, 11, 13], [3, 5, 7, 9, 11, 13, 15]]

print(f"{'Window Size':24} | {'<=10 Count':>10} | {'Total':>8} | {'Ratio (%)':>10}")
print('-'*48)

for window_size in window_sizes:
    filename = f'window size/{file_name}{window_size}.csv'
    try:
        df = pd.read_csv(filename)
        # 排除NaN
        valid = df['velocity_magnitude'].dropna()
        count_le_10 = (valid <= 10).sum()
        total = valid.shape[0]
        ratio = (count_le_10 / total * 100) if total > 0 else 0
        print(f"{str(window_size):24} | {count_le_10:10} | {total:8} | {ratio:10.2f}")
    except FileNotFoundError:
        print(f"{filename} not found.") 