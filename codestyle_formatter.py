# do not move
# this file has to be in the root folder of the project

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

clang_format_file_path = "./.clang-format"
allowed_file_extensions = [".c", ".cpp", ".h", ".hpp"]
clang_format_executable = "clang-format"
clang_format_arguments = "-style=file"
lines_to_comment_out_prefixes = ["#pragma", "#ifdef", "#ifndef", "#undef", "#if", "#elif", "#else", "#endif"]
comment_string = "//123@123"
folders_to_ignore = ["cmake-build-debug", "tsl"]
files_to_ignore = []


def get_files_to_format(directory_path):
    result = []
    for path, subdirs, files in os.walk(directory_path, topdown=True):
        subdirs[:] = [d for d in subdirs if d not in folders_to_ignore]
        for name in files:
            ignore_this_file = False
            for forbidden_file_name in files_to_ignore:
                if name.endswith(forbidden_file_name):
                    ignore_this_file = True
            if ignore_this_file:
                continue
            cur_file_path = os.path.join(path, name)
            for allowed_extension in allowed_file_extensions:
                if cur_file_path.endswith(allowed_extension):
                    result.append(cur_file_path)
                    break
    return result


def preprocess_file(target_file_path):
    with open(target_file_path, 'r') as file:
        file_data = file.read()

    for line_to_comment_out_prefix in lines_to_comment_out_prefixes:
        file_data = file_data.replace(line_to_comment_out_prefix, comment_string + line_to_comment_out_prefix)

    with open(target_file_path, 'w') as file:
        file.write(file_data)


def undo_preprocessing_file(target_file_path):
    with open(target_file_path, 'r') as file:
        file_data = file.read()

    for line_to_comment_out_prefix in lines_to_comment_out_prefixes:
        file_data = file_data.replace(comment_string + line_to_comment_out_prefix, line_to_comment_out_prefix)

    with open(target_file_path, 'w') as file:
        file.write(file_data)


def print_clang_format_diff(target_file_path, safe_diff):
    if not safe_diff:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            temp_file_name = shutil.copy(target_file_path, tmp_dir_name)
            shutil.copy(clang_format_file_path, tmp_dir_name)
            preprocess_file(temp_file_name)
            subprocess.run([clang_format_executable, clang_format_arguments, "--dry-run", temp_file_name])
    else:
        subprocess.run([clang_format_executable, clang_format_arguments, "--dry-run", target_file_path])


def apply_clang_format(target_file_path, safe_apply):
    if not safe_apply:
        preprocess_file(target_file_path)
    subprocess.run([clang_format_executable, clang_format_arguments, "-i", target_file_path])
    if not safe_apply:
        undo_preprocessing_file(target_file_path)


def main():
    parser = argparse.ArgumentParser(description="Apply or check clang-format CodeStyle rules to the whole project")
    parser.add_argument("-c", "--check", help="Check project files for CodeStyle issues",
                        action="store_true")

    parser.add_argument("-a", "--apply", help="Apply .clang-format CodeStyle to all project files",
                        action="store_true")

    parser.add_argument("-s", "--safe", help="Do check and apply without preprocessing",
                        action="store_true")

    parser.add_argument("-q", "--quiet", help="No debug output, so the runner could easily interpret output",
                        action="store_true")

    args = parser.parse_args()

    if not os.path.exists(clang_format_file_path):
        print("ERROR: .clang-format file not found")
        sys.exit(1)

    files_to_format_paths = get_files_to_format(os.getcwd())
    for file_to_format_path in files_to_format_paths:
        if args.check:
            if not args.quiet:
                print("=" * 99 + "\n" + f"Problems with file: {file_to_format_path}:")
            print_clang_format_diff(file_to_format_path, args.safe)
        if args.apply:
            if not args.quiet:
                print(f"Applied format to the file: {file_to_format_path}")
            apply_clang_format(file_to_format_path, args.safe)


if __name__ == "__main__":
    main()