import project_setup


path = project_setup.find_project_root()

print(f"Project root found at: {path}")

data_path_raw = project_setup.data_path("example_data.csv", stage="raw")

print(f"Raw data path: {data_path_raw}")