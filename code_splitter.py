import re
import random
import os
import glob
import json

def get_functions_code(script_code):
    code_lines = script_code.split("\n")
    functions = []

    function_pattern = re.compile(r"^(def\s+\w+\(.*?\):)", re.MULTILINE)
    matches = list(function_pattern.finditer(script_code))

    for i in range(len(matches)):
        start_line_number = script_code.count('\n', 0, matches[i].start())
        if i + 1 < len(matches):
            end_line_number = script_code.count('\n', 0, matches[i + 1].start())
            function_lines = code_lines[start_line_number:end_line_number]
        else:
            function_lines = code_lines[start_line_number:]
        functions.append("\n".join(function_lines))  
    
    return functions

def split_function_code(function_code):
    lines = [line for line in function_code.strip().splitlines() if line.strip()]
    total_lines = len(lines)
    
    if total_lines < 4:
        return None

    middle_min_start = 2
    middle_max_end = total_lines - 1
    middle_max_length = int(total_lines * 0.25)
    
    if (middle_max_end - middle_min_start) >= 2:
        middle_random_length = random.randint(1, middle_max_length) if middle_max_length != 1 else middle_max_length
        middle_start = random.randint(middle_min_start, int(total_lines * 0.75))
        middle_end = middle_start + middle_random_length
    else:
        middle_start = middle_min_start
        middle_end = middle_max_end
     
    prefix_str = "\n".join(lines[:middle_start])
    middle_str = "\n".join(lines[middle_start:middle_end])
    suffix_str = "\n".join(lines[middle_end:])
    
    return prefix_str, middle_str, suffix_str

def main(source_dir, output_json):
    dataset = []
    all_python_files_paths = glob.glob(os.path.join(source_dir, "*", "*.py"))
    for file in all_python_files_paths:
        with open(file, 'r') as f:
            python_code = f.read()
        functions = get_functions_code(python_code)
        for i, function_code in enumerate(functions, start=1):
            splitted_function = split_function_code(function_code)
            if splitted_function:
                prefix, middle, suffix = splitted_function
                dataset.append({"prefix": prefix, "middle": middle, "suffix": suffix})

    with open(output_json, "w") as json_file:
        json.dump(dataset, json_file, indent=4)

if __name__ == "__main__":
    source_dir = "repository_python_files/"
    output_file = "splitted_code.json"
    main(source_dir, output_file)