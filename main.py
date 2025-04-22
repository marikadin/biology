from cd_ailib import ColorAnalyzer
from csv_graphlib import pHAnalyzer
import numpy as np
import os
 


def csv_analysis():
    csv_analyzer = pHAnalyzer('spiroline.xlsx')
    csv_analyzer.plot()

def color_analysis():
    parent_folder = 'crops'
    # Get model type from user
    while True:
        model_type = input("Which model type would you like to use? (lstm/linear): ").lower()
        if model_type in ['lstm', 'linear']:
            break
        print("Invalid choice. Please enter 'lstm' or 'linear'.")

    # Get folder selection from user
    print("\nAvailable folders in parent directory:")
    available_folders = [f.name for f in os.scandir(parent_folder) if f.is_dir()]
    for i, folder in enumerate(available_folders, 1):
        print(f"{i}. {folder}")
    
    print("\nSelect folders (enter numbers separated by spaces, or 'all' for all folders):")
    while True:
        selection = input("> ").strip()
        if selection.lower() == 'all':
            selected_folders = available_folders
            break
        try:
            indices = [int(x) - 1 for x in selection.split()]
            if all(0 <= i < len(available_folders) for i in indices):
                selected_folders = [available_folders[i] for i in indices]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter numbers or 'all'.")

    color_analyzer = ColorAnalyzer(target_color=np.array([0, 60, 0]))
    color_analyzer.analyze_folders(parent_folder, selected_folders, model_type=model_type)

def main():
    while True:
        print("\nWhich analysis would you like to run?")
        print("1. CSV/pH Analysis")
        print("2. Color Analysis")
        print("3. Both")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            csv_analysis()
        elif choice == '2':
            color_analysis()
        elif choice == '3':
            csv_analysis()
            color_analysis()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()