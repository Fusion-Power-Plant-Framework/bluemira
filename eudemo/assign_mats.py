import csv
from pathlib import Path

fp_p = Path(__file__).resolve().parent

# Specify the path to your CSV file
csv_file = fp_p / "EUDEMO-geometry.csv"

comp_name_to_mat_mapping = {
    "VacuumVessel": "mat_1",
    "Divertor": "mat_2",
    "Blanket": "mat_3",
    "Thermal_Shield": "mat_4",
    "TFCoil": "mat_5",
    "Poloidal_Coils": "mat_6",
    "Cryostat": "mat_7",
    "RadiationShield": "mat_8",
}

mat_list = []

# Open the CSV file
with open(csv_file) as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        comp_name = row[1]
        mat_name = comp_name_to_mat_mapping[comp_name]
        mat_list.append(mat_name)

mat_file_name = "EUDEMO_materials.txt"

# Open the file in write mode
with open(Path.cwd() / mat_file_name, "w") as file:
    # Write your content to the file
    for i, mat_name in enumerate(mat_list):
        file.write(mat_name)
        if i != len(mat_list) - 1:
            file.write("\n")

# Confirm that the file has been written
print(
    f"File '{mat_file_name}' has been written successfully. {len(mat_list)} materials written."
)
