import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_excel('spiroline.xlsx')

data.columns = data.columns.str.strip()
data['_datetime'] = pd.to_datetime(data['_sample_time'])

tubes = data['_sample_tube_title'].unique().tolist()

dicti = {'Shefer 3 - 1%': ('26E5', '4527'), 'Shefer 3 - 0.5%': ('4092', '26T8'), 'Living - 0.5%': ('2854', '4578'),
         'Living - 1%': ('44F3', '4595')}

reverse_dict = {tube_id: sample_name for sample_name, tube_ids in dicti.items() for tube_id in tube_ids}


def plot_samples_and_avg(sample_tube_title_1, sample_tube_title_2, time_jump_minutes, month_range):
    sample_data_1 = data[data['_sample_tube_title'] == sample_tube_title_1]
    sample_data_2 = data[data['_sample_tube_title'] == sample_tube_title_2]

    start_month, end_month = map(int, month_range.split('-'))
    sample_data_1 = sample_data_1[
        (sample_data_1['_datetime'].dt.month >= start_month) & (sample_data_1['_datetime'].dt.month <= end_month)]
    sample_data_2 = sample_data_2[
        (sample_data_2['_datetime'].dt.month >= start_month) & (sample_data_2['_datetime'].dt.month <= end_month)]

    sample_data_1 = sample_data_1.iloc[::int(time_jump_minutes / 30)]
    sample_data_2 = sample_data_2.iloc[::int(time_jump_minutes / 30)]

    merged_data = pd.merge(sample_data_1[['_datetime', '_ph_value']],
                           sample_data_2[['_datetime', '_ph_value']],
                           on='_datetime',
                           suffixes=('_1', '_2'))

    merged_data['avg_ph'] = merged_data[['_ph_value_1', '_ph_value_2']].mean(axis=1)

    # Convert datetime to numerical timestamps for linear fitting
    # Convert datetime to numerical timestamps
    x = merged_data['_datetime'].map(pd.Timestamp.timestamp).values
    y = merged_data['avg_ph'].values

    # Center x-values around the mean
    x_centered = x - x.mean()

    # Fit a linear model
    coeffs = np.polyfit(x_centered, y, 1)  # Linear fit (1st degree polynomial)
    linear_fit = np.poly1d(coeffs)
    merged_data['linear_avg_ph'] = linear_fit(x_centered)

    # Extract slope (m) and intercept (b)
    m, b = coeffs  # Now b represents pH at mean timestamp

    # Format slope in scientific notation if necessary
    m_str = f"{m:.2e}" if abs(m) < 0.001 or abs(m) > 1000 else f"{m:.10f}"

    if b > 0:
        equation_text = f"y = {m_str}x + {b:.2f}"
    elif b < 0:
        equation_text = f"y = {m_str}x - {abs(b):.2f}"
    else:
        equation_text = f"y = {m_str}x"
    key1 = next((k for k, v in dicti.items() if sample_tube_title_1 in v), None)
    key2 = next((k for k, v in dicti.items() if sample_tube_title_2 in v), None)
    # Plot data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(sample_data_1['_datetime'], sample_data_1['_ph_value'], marker='o', linestyle='-',
            label=f'{key1}: {sample_tube_title_1}')
    ax.plot(sample_data_2['_datetime'], sample_data_2['_ph_value'], marker='x', linestyle='-',
            label=f'{key2}: {sample_tube_title_2}')
    ax.plot(merged_data['_datetime'], merged_data['avg_ph'], marker='o', linestyle='-', color='green', label='Avg pH')
    ax.plot(merged_data['_datetime'], merged_data['linear_avg_ph'], color='red', linestyle='--', linewidth=2,
            label='Linear Trendline')

    # Add equation text to the plot
    ax.text(0.05, 0.9, equation_text, transform=ax.transAxes, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

    # Formatting
    ax.set_title('pH Values Over Time')
    ax.set_xlabel('Date and Time')
    ax.set_ylabel('pH Value')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(rotation=45)
    ax.grid(True)
    ax.legend()
    plt.show()


def execute():
    whichone = input("Would you like to input samples manually? (yes/no): ")

    if whichone.lower() == "yes":
        print("Available tube IDs:", tubes)
        sample_tube_title_1 = input("Enter the first sample tube ID: ")
        while sample_tube_title_1 not in tubes:
            sample_tube_title_1 = input("Invalid. Enter a valid tube ID: ")

        sample_tube_title_2 = input("Enter the second sample tube ID: ")
        while sample_tube_title_2 not in tubes or sample_tube_title_2 == sample_tube_title_1:
            sample_tube_title_2 = input("Invalid or duplicate. Enter a different valid tube ID: ")

    else:
        sample_tube_title_1, sample_tube_title_2 = tubes[:2]  # Automatically select first two
        print(f"Selected Tubes: {sample_tube_title_1}, {sample_tube_title_2}")

    time_jump_minutes = 30
    month_range = "1-2"
    plot_samples_and_avg(sample_tube_title_1, sample_tube_title_2, time_jump_minutes, month_range)


if __name__ == "__main__":
    execute()