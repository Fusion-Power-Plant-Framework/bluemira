import os

import mypy.api


def run_mypy(filename):
    result, _, _ = mypy.api.run([filename])
    return result if result else ""


def filter_error_messages(results, error_message):
    filtered_results = [line for line in results.split("\n") if error_message in line]
    return "\n".join(filtered_results)


def process_files_in_directory(directory, error_message=None):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filename = os.path.join(root, file)
                mypy_results = run_mypy(filename)
                if error_message is not None:
                    filtered_results = filter_error_messages(mypy_results, error_message)
                else:
                    filtered_results = mypy_results

                if filtered_results:
                    print(f"File: {filename}\n{filtered_results}\n{'-' * 50}")

        """if dirs:
            user_input = input(f"Do you want to process files in the next directory? (y/n): ").lower()
            if user_input != 'y':
                break"""


if __name__ == "__main__":
    target_directory = "bluemira/materials"  # Replace with your actual directory
    # error_message = 'Function is missing a return type annotation'
    error_message = "[assignment]"
    process_files_in_directory(target_directory, error_message)
