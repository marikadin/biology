import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class pHAnalyzer:
    def __init__(self, excel_file_path):
        self.data = pd.read_excel(excel_file_path)
        self.data.columns = self.data.columns.str.strip()
        self.data['_datetime'] = pd.to_datetime(self.data['_sample_time'])
        self.tubes = self.data['_sample_tube_title'].unique().tolist()
        
        # Default sample mapping dictionary
        self.sample_mapping = {
            'Shefer_1%': ('26E5', '4527'),
            'Shefer_0.5%': ('4092', '26T8'),
            'Living_0.5%': ('2854', '4578'),
            'Living_1%': ('44F3', '4595')
        }
        self.reverse_mapping = {
            tube_id: sample_name 
            for sample_name, tube_ids in self.sample_mapping.items() 
            for tube_id in tube_ids
        }

def plot_samples_and_avg(self, sample_tube_title_1, sample_tube_title_2, time_jump_minutes=30, month_range="1-12"):
    """
    Plot pH values and average for two samples over time.
    
    Args:
        sample_tube_title_1 (str): First tube ID
        sample_tube_title_2 (str): Second tube ID
        time_jump_minutes (int): Time interval between data points
        month_range (str): Range of months to plot (format: "start-end")
    """
    # Validate that the tube IDs exist in the data
    if sample_tube_title_1 not in self.tubes or sample_tube_title_2 not in self.tubes:
        print(f"Error: One or both tube IDs not found in data")
        print(f"Tube 1: {sample_tube_title_1} {'(found)' if sample_tube_title_1 in self.tubes else '(not found)'}")
        print(f"Tube 2: {sample_tube_title_2} {'(found)' if sample_tube_title_2 in self.tubes else '(not found)'}")
        print(f"Available tubes: {self.tubes}")
        return

    # Get the data for each sample
    sample_data_1 = self.data[self.data['_sample_tube_title'] == sample_tube_title_1].copy()
    sample_data_2 = self.data[self.data['_sample_tube_title'] == sample_tube_title_2].copy()

    # Print data validation
    print(f"\nData points found:")
    print(f"Tube {sample_tube_title_1}: {len(sample_data_1)} points")
    print(f"Tube {sample_tube_title_2}: {len(sample_data_2)} points")

    start_month, end_month = map(int, month_range.split('-'))
    sample_data_1 = sample_data_1[
        (sample_data_1['_datetime'].dt.month >= start_month) & 
        (sample_data_1['_datetime'].dt.month <= end_month)
    ]
    sample_data_2 = sample_data_2[
        (sample_data_2['_datetime'].dt.month >= start_month) & 
        (sample_data_2['_datetime'].dt.month <= end_month)
    ]

    # Print filtered data points
    print(f"\nData points after month filtering ({month_range}):")
    print(f"Tube {sample_tube_title_1}: {len(sample_data_1)} points")
    print(f"Tube {sample_tube_title_2}: {len(sample_data_2)} points")

    sample_data_1 = sample_data_1.iloc[::int(time_jump_minutes / 30)]
    sample_data_2 = sample_data_2.iloc[::int(time_jump_minutes / 30)]

    merged_data = pd.merge(
        sample_data_1[['_datetime', '_ph_value']],
        sample_data_2[['_datetime', '_ph_value']],
        on='_datetime',
        suffixes=('_1', '_2')
    )

    # Print merge results
    print(f"\nMerged data points: {len(merged_data)} points")

    merged_data['avg_ph'] = merged_data[['_ph_value_1', '_ph_value_2']].mean(axis=1)

    x = merged_data['_datetime'].map(pd.Timestamp.timestamp).values
    y = merged_data['avg_ph'].values
    x_centered = x - x.mean()

    coeffs = np.polyfit(x_centered, y, 1)
    linear_fit = np.poly1d(coeffs)
    merged_data['linear_avg_ph'] = linear_fit(x_centered)

    m, b = coeffs
    m_str = f"{m:.2e}" if abs(m) < 0.001 or abs(m) > 1000 else f"{m:.10f}"

    if b > 0:
        equation_text = f"y = {m_str}x + {b:.2f}"
    elif b < 0:
        equation_text = f"y = {m_str}x - {abs(b):.2f}"
    else:
        equation_text = f"y = {m_str}x"

    # Get sample names from mapping
    key1 = next((k for k, v in self.sample_mapping.items() if sample_tube_title_1 in v), sample_tube_title_1)
    key2 = next((k for k, v in self.sample_mapping.items() if sample_tube_title_2 in v), sample_tube_title_2)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot individual samples
    ax.plot(sample_data_1['_datetime'], sample_data_1['_ph_value'], 
            marker='o', linestyle='-', label=f'{key1}: {sample_tube_title_1}')
    ax.plot(sample_data_2['_datetime'], sample_data_2['_ph_value'], 
            marker='x', linestyle='-', label=f'{key2}: {sample_tube_title_2}')
    
    # Plot average and trendline
    ax.plot(merged_data['_datetime'], merged_data['avg_ph'], 
            marker='o', linestyle='-', color='green', label='Avg pH')
    ax.plot(merged_data['_datetime'], merged_data['linear_avg_ph'], 
            color='red', linestyle='--', linewidth=2, label='Linear Trendline')

    # Add equation text
    ax.text(0.05, 0.9, equation_text, transform=ax.transAxes, fontsize=12, 
            color='red', bbox=dict(facecolor='white', alpha=0.5))

    # Configure plot appearance
    ax.set_title('pH Values Over Time')
    ax.set_xlabel('Date and Time')
    ax.set_ylabel('pH Value')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(rotation=45)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    def get_sample_inputs(self):
        """Get sample inputs from user"""
        print("Available tube IDs:", self.tubes)
        sample_tube_title_1 = input("Enter the first sample tube ID: ")
        while sample_tube_title_1 not in self.tubes:
            sample_tube_title_1 = input("Invalid. Enter a valid tube ID: ")

        sample_tube_title_2 = input("Enter the second sample tube ID: ")
        while sample_tube_title_2 not in self.tubes or sample_tube_title_2 == sample_tube_title_1:
            sample_tube_title_2 = input("Invalid or duplicate. Enter a different valid tube ID: ")
            
        return sample_tube_title_1, sample_tube_title_2

    def get_default_samples(self):
        """Get first two samples as default"""
        sample_tube_title_1, sample_tube_title_2 = self.tubes[:2]
        print(f"Selected Tubes: {sample_tube_title_1}, {sample_tube_title_2}")
        return sample_tube_title_1, sample_tube_title_2

    def plot(self, manual_input=None, time_jump_minutes=30, month_range="1-12"):
        """
        Plot samples based on user input or default values
        
        Args:
            manual_input (bool, optional): If None, will prompt user. If True, manual input. If False, use defaults.
            time_jump_minutes (int): Time interval between data points
            month_range (str): Range of months to plot (format: "start-end")
        """
        if manual_input is None:
            whichone = input("Would you like to input samples manually? (yes/no): ")
            manual_input = whichone.lower() == "yes"

        if manual_input:
            sample_tube_title_1, sample_tube_title_2 = self.get_sample_inputs()
        else:
            sample_tube_title_1, sample_tube_title_2 = self.get_default_samples()

        self.plot_samples_and_avg(sample_tube_title_1, sample_tube_title_2, time_jump_minutes, month_range)

def main():
    analyzer = pHAnalyzer('spiroline.xlsx')
    analyzer.plot()

if __name__ == "__main__":
    main()