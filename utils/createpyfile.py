# -*- coding: utf-8 -
"""Module-level docstring"""
import sys


def create_python_file(file_name):
    """Create a new Python file with the given filename."""
    # Check if the filename is provided
    if not file_name:
        print("Error: No filename provided.")
        sys.exit(1)

    # Create the file with the given filename
    try:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write("# -*- coding: utf-8 -*-\n")
            file.write('"""Module-level docstring"""\n')
        print(f"File '{file_name}' created successfully with the header.")
    except FileNotFoundError as e:
        print(f"Error creating file '{file_name}': {e}")


if __name__ == "__main__":
    # Check if the filename is provided
    if len(sys.argv) != 2:
        print("Usage: python createpyfile.py <filename>")
        sys.exit(1)

    # Get the filename from the command line argument
    filename = sys.argv[1]
    filename_with_extension = filename + ".py"
    create_python_file(filename_with_extension)
