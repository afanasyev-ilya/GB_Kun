import os
import sys
import argparse
import subprocess


clang_format_file_path = "./.clang-format"
allowed_file_extensions = [".c", ".cpp", ".h", ".hpp"]
clang_format_executable = "clang-format"
clang_format_arguments = "-style=file"


def get_files_to_format(directory_path):
    result = []
    for path, subdirs, files in os.walk(directory_path):
        for name in files:
            cur_file_path = os.path.join(path, name)
            for allowed_extension in allowed_file_extensions:
                if cur_file_path.endswith(allowed_extension):
                    result.append(cur_file_path)
                    break
    return result


def get_clang_format_diff(target_file_path):
    result = subprocess.check_output([clang_format_executable,
                                      clang_format_arguments, "--dry-run", target_file_path], stderr=STDOUT)
    return result.replace("\\n", "\n")


def apply_clang_format(target_file_path):
    subprocess.run([clang_format_executable, clang_format_arguments, "-i", target_file_path])


def main():
    parser = argparse.ArgumentParser(description="Apply or check clang-format CodeStyle rules to the whole project")
    parser.add_argument("-c", "--check", help="Check project files for CodeStyle issues",
                        action="store_true")

    parser.add_argument("-a", "--apply", help="Apply .clang-format CodeStyle to all project files",
                        action="store_true")

    args = parser.parse_args()

    if not os.path.exists(clang_format_file_path):
        print("ERROR: .clang-format file not found")
        sys.exit(1)

    files_to_format_paths = get_files_to_format(os.getcwd())
    for file_to_format_path in files_to_format_paths:
        if args.check:
            check_result = get_clang_format_diff(file_to_format_path)
            if check_result:
                print(f"Found problems with file: {file_to_format_path}:\n{check_result}0\n")
        if args.apply:
            print(f"Applied format to the file: {file_to_format_path}")
            apply_clang_format(file_to_format_path)


if __name__ == "__main__":
    main()
