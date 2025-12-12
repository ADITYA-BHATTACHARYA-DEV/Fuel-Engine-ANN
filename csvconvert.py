import pandas as pd
import os
import glob
import sys


def convert_excel_to_csv(source_folder):
    """
    Recursively finds all Excel files (.xls, .xlsx) in the source_folder
    and all its subdirectories. Converts every sheet in each file
    to a separate CSV file.

    The CSV files are saved in a parallel folder structure under
    a 'converted_csvs' directory at the same level as the source folder.
    """

    # Define the base output folder
    output_base = os.path.join(source_folder, "converted_csvs")

    # Create the output base folder if it doesn’t exist
    os.makedirs(output_base, exist_ok=True)

    # Search recursively for Excel files
    search_path_xlsx = os.path.join(source_folder, "**", "*.xlsx")
    search_path_xls = os.path.join(source_folder, "**", "*.xls")

    excel_files = glob.glob(search_path_xlsx, recursive=True) + glob.glob(search_path_xls, recursive=True)

    if not excel_files:
        print(f"No Excel files found in the folder (or subfolders): {source_folder}")
        return

    print(f"Found {len(excel_files)} Excel file(s). Starting conversion...\n")

    total_sheets_converted = 0

    for file_path in excel_files:
        try:
            # Derive relative path (to replicate folder structure)
            rel_path = os.path.relpath(file_path, source_folder)
            rel_dir = os.path.dirname(rel_path)

            # Create target folder inside 'converted_csvs' with same structure
            target_dir = os.path.join(output_base, rel_dir)
            os.makedirs(target_dir, exist_ok=True)

            # Get base name of file (without extension)
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Read all sheet names
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names

            print(f"--- Processing '{os.path.basename(file_path)}' ---")
            print(f"    Found {len(sheet_names)} sheet(s)")

            for sheet_name in sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)

                # Make unique CSV name
                if len(sheet_names) > 1:
                    csv_name = f"{base_name}_{sheet_name}.csv"
                else:
                    csv_name = f"{base_name}.csv"

                csv_path = os.path.join(target_dir, csv_name)

                # Write CSV
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"  > Saved '{csv_name}' to '{target_dir}'")

                total_sheets_converted += 1

        except Exception as e:
            print(f"  [!] Error processing '{os.path.basename(file_path)}': {e}")

    print("\n--- Conversion Complete ---")
    print(f"Total Excel files processed: {len(excel_files)}")
    print(f"Total sheets converted: {total_sheets_converted}")
    print(f"All CSVs saved under: {output_base}")


def main():
    """
    Main entry point — prompts the user for a folder path and starts conversion.
    """
    folder_path = input("Enter the full path to the folder containing your Excel files: ").strip()

    if not os.path.isdir(folder_path):
        print("\nError: The provided path is not a valid directory.")
        print("Please run the script again and enter a correct path.")

        if "\\" in folder_path:
            print("\nPython Tip: If your path has backslashes (\\), try using r'...' to avoid escape issues.")
            print(r"Example: r'F:\KASHYAP PROJECT\Fuel_Engine\.venv\Ammonia'")
        sys.exit(1)

    convert_excel_to_csv(folder_path)


if __name__ == "__main__":
    main()
