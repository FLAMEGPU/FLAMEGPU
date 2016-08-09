"""
Script to create full archive including executables for a release.
All files required to build and run examples should be included, but not any temporary files
"""
import os 
import re
import glob
import shutil
import argparse
import errno
import zipfile


# Default output filename
DEFAULT_OUTPUT = "FLAME-GPU-SDK.zip"

# Relative path to root directory from this source file, enabling command line usage from any location.
RELATIVE_ROOT_DIR = "../"

# List of regular expressions for files to be ignored from the archive.
# Patterns are evaluated against the final destination in the archive (i.e. bin/ not ../bin)
EXCLUDE_PATTERNS = [
    '^.*(.\git).*$',
    '^(bin).*(Debug_).*(?<!dll)$',
    # '^(examples).*(dynamic).*\.(cu|h)$',
    '^(examples).*(x64).*$',
    '^(examples).*\.(sdf|opensdf|suo|pdb)$',
    '^.*\.(zip)$'
]

def run(args):
    # Get the location of this file.
    script_loc = os.path.dirname(os.path.realpath(__file__))

    # Create the archive in the working directory this was called from.
    zipf = zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED)

    # Change directory to the location of the script to simplify population of zip file.
    os.chdir(script_loc)
    
    # Compound individual exclusion regular expressions into single compiled regex
    exclude_pattern = "|".join(EXCLUDE_PATTERNS)
    exclude_re = re.compile(exclude_pattern)

    # For each file in the file hierarchy
    for root, dirs, files in os.walk(RELATIVE_ROOT_DIR):
        for file in files:
            # Get file location on disk
            input_loc = os.path.normpath(os.path.join(root, file))
            # Get relative location within archive
            output_loc = os.path.normpath(os.path.relpath(input_loc, RELATIVE_ROOT_DIR))
            # Check if the file matches the exclude list
            exclude_match = exclude_re.match(output_loc)
            if not exclude_match:
                # Add to zip
                zipf.write(input_loc, output_loc)
                print(output_loc)

    # Close the archive
    zipf.close()
    


def main():
    # Argument parsing.
    parser = argparse.ArgumentParser(description="Create an archive of the required files for standalone execution of examples")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output Directory for archive. Defaults to {:}".format(DEFAULT_OUTPUT),
        default=DEFAULT_OUTPUT
        )

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
