import os
import re
import random

# Define the directory path
directory_path = "/Users/natanan/Documents/GitHub/Powerful-N-Gram-Model-Distillation-of-Thai-Language-Model-with-N-Gram/Datasets"

# Function to extract [str_content] from a random subset of shuffled files and record file names
def extract_str_content_random_and_record_filenames(directory, num_files):
    # Define the regular expression pattern
    pattern = r'<doc id="[\d]+" url="[^"]+" title="[^"]+">(.+?)</doc>'
    compiled_pattern = re.compile(pattern, re.DOTALL)

    # Get a list of all files in the directory
    all_files = [filename for filename in os.listdir(directory) if filename.endswith(".txt")]

    # Shuffle the list of files randomly
    random.shuffle(all_files)

    # Select the first num_files files for processing
    selected_files = all_files[:num_files]

    # Initialize an empty string to store the combined [str_content]
    combined_str_content = ''

    # Initialize an empty list to store file names
    used_file_names = []

    # Iterate through selected files
    for filename in selected_files:
        file_path = os.path.join(directory, filename)
        used_file_names.append(filename)  # Record the used file name

        # Open the file and read its content
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

            # Use regular expression to extract [str_content]
            str_contents = compiled_pattern.findall(file_content)

            # Append extracted [str_content] to combined_str_content
            for content in str_contents:
                combined_str_content += content + '\n\n'  # Add a newline for separation

    # Write the combined [str_content] to a new file
    output_file_path = "/Users/natanan/Documents/GitHub/Powerful-N-Gram-Model-Distillation-of-Thai-Language-Model-with-N-Gram/Datasets/extracted_str_content.txt"
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(combined_str_content)

    # Write the used file names to a separate file
    used_files_path = "/Users/natanan/Documents/GitHub/Powerful-N-Gram-Model-Distillation-of-Thai-Language-Model-with-N-Gram/Datasets/used_files.txt"
    with open(used_files_path, 'w', encoding='utf-8') as used_files:
        for file_name in used_file_names:
            used_files.write(file_name + '\n')

    print("Extracted [str_content] from random files and written to", output_file_path)
    print("Used file names written to", used_files_path)

# Define the number of files to process randomly
num_files_to_process = 30000

# Call the function to extract [str_content] from a random subset of files and record file names
extract_str_content_random_and_record_filenames(directory_path, num_files_to_process)
