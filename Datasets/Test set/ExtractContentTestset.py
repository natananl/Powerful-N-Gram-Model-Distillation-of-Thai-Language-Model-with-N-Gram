import os
import re
import random
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openthaigpt/openthaigpt-1.0.0-beta-7b-chat-ckpt-hf", use_fast=False)

# Define the directory path
directory_path = "/Users/natanan/Documents/GitHub/Powerful-N-Gram-Model-Distillation-of-Thai-Language-Model-with-N-Gram/Datasets/documents-nsc"

# Function to extract [str_content] from a random subset of shuffled files and record file names
def extract_str_content_random_and_record_filenames(directory, used_files):
    # Define the regular expression pattern
    pattern = r'<doc id="[\d]+" url="[^"]+" title="[^"]+">(.+?)</doc>'
    compiled_pattern = re.compile(pattern, re.DOTALL)

    # Get a list of all files in the directory
    all_files = [filename for filename in os.listdir(directory) if filename.endswith(".txt")]

    # Shuffle the list of files randomly
    random.shuffle(all_files)

    # Remove all used files
    unused_files = [file for file in all_files if file not in used_files]

    # Initialize an empty string to store the combined [str_content]
    combined_str_content = ''

    # Initialize an empty list to store file names
    used_file_names = []

    # Store total words
    total_token = 0

    # Iterate through selected files
    for filename in unused_files:
        file_path = os.path.join(directory, filename)
        

        # Open the file and read its content
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

            # Use regular expression to extract [str_content]
            str_contents = compiled_pattern.findall(file_content)

            # Iterate through each content and count tokens
            for content in str_contents:
                token_count = len(tokenizer.tokenize(content))
                if total_token + token_count < 10000:
                    total_token += token_count
                    combined_str_content += content + '\n\n'  # Add a newline for separation
                    used_file_names.append(filename)  # Record the used file name
                else:
                    break  # Stop adding content if the token count exceeds 10000

        if total_token >= 10000:
            break  # Stop iterating through files if the token count exceeds 10000

    # Write the combined [str_content] to a new file
    output_file_path = os.path.join("/Users/natanan/Documents/GitHub/Powerful-N-Gram-Model-Distillation-of-Thai-Language-Model-with-N-Gram/Datasets/Test set", "extracted_str_content.txt")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(combined_str_content)

    # Write the used file names to a separate file
    used_files_path = os.path.join("/Users/natanan/Documents/GitHub/Powerful-N-Gram-Model-Distillation-of-Thai-Language-Model-with-N-Gram/Datasets/Test set", "used_files_for_test.txt")
    with open(used_files_path, 'w', encoding='utf-8') as used_files:
        for file_name in used_file_names:
            used_files.write(file_name + '\n')

    print("Extracted [str_content] from random files and written to", output_file_path)
    print("Used file names written to", used_files_path)

# Used files list
used_files_path = "/Users/natanan/Documents/GitHub/Powerful-N-Gram-Model-Distillation-of-Thai-Language-Model-with-N-Gram/Datasets/Training for N-gram/used_files.txt"
with open(used_files_path) as f:
    used_files = {name.strip() for name in f}

# Define the number of files to process
num_files_to_process = 30000

# Call the function to extract [str_content] from a random subset of files and record file names
extract_str_content_random_and_record_filenames(directory_path, used_files)

