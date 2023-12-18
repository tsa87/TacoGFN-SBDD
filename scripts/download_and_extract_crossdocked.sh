# Download and extract the crossdocked dataset.

file_url="https://www.dropbox.com/s/h40wvg5nsoo7lqu/crossdocked_pocket10.tar.gz?dl=1"
file_name="crossdocked_pocket10.tar.gz"
dataset_directory="dataset/"

# Full path for the file including the extract directory
full_file_path="${dataset_directory}${file_name}"

# Check if the file already exists
if [ -f "$full_file_path" ]; then
    echo "File already exists. Skipping download and extraction."
else
    # Create the extract directory if it doesn't exist
    mkdir -p "$dataset_directory"

    # Download the file directly into the extract directory
    wget -O "$full_file_path" "$file_url"

    # Extract the file
    tar -xzf "$full_file_path" -C "$dataset_directory"
fi