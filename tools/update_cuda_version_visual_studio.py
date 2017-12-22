#! /bin/python

"""
Simple script to update the CUDA version in a visual studio project

@author Peter Heywood <p.heywood@sheffield.ac.uk>
"""


import argparse
import sys
import os
import shutil
import distutils.dir_util
import re

RELATIVE_EXAMPLES_DIR="../examples"

VCXPROJ_EXT = ".vcxproj"
PROJECT_FILE_FMT = "{:}.vcxproj"

VERSION_FORMAT = "^[0-9]+\\.[0-9]+$"

REPLACEMENTS = [
    {"pattern": "(^.*BuildCustomizations\\\\CUDA )([0-9]+\\.[0-9]+)(\\.targets[\\s\\S]*)$", "group": 2, "regex": None},
    {"pattern": "(^.*BuildCustomizations\\\\CUDA )([0-9]+\\.[0-9]+)(\\.props[\\s\\S]*)$", "group": 2, "regex": None}
]

def checkVersion(version):
    regex = re.compile(VERSION_FORMAT)
    if not regex.match(version):
        return False
    else:
        return True

def examplesDirExists(examples_dir):
    # Check if the examples directory exists.
    return os.path.isdir(examples_dir)

def findAllVcxprojFiles(examples_dir):
    names = []

    # Find all possible examples
    examples = os.listdir(examples_dir)
    # For each example, list files.
    for example in examples:
        # Look for the file
        expected_vcxproj_filename = PROJECT_FILE_FMT.format(example)
        expected_vcxproj_path = os.path.join(examples_dir, os.path.join(example, expected_vcxproj_filename))
        if os.path.isfile(expected_vcxproj_path):
            names.append(example)

    return names


def updateCudaVersionInVsproj(name, cuda_version, examples_dir):

    for d in REPLACEMENTS:
        if d["regex"] is None:
            d["regex"] = re.compile(d["pattern"])

    # Get the path to the project file.
    example_dir = os.path.join(examples_dir, name)
    # Check the example dir exists.
    if os.path.isdir(example_dir):
        example_vcxproj = os.path.join(example_dir, PROJECT_FILE_FMT.format(name))
        # Check it exists
        if os.path.isfile(example_vcxproj):
            # If the file exist - do something.

            content = None
            try:
                with open(example_vcxproj, "r") as file:
                    content = file.readlines()
            except Exception as e: 
                print("Exception whilst reading {:}\n > {:}".format(fpath, e))
                return False

            try:
                with open(example_vcxproj, "w") as file:
                    for line in content:
                        newline = line
                        for d in REPLACEMENTS:
                            match = d["regex"].match(line)
                            if match:
                                newline = ""
                                for i, group in enumerate(match.groups()):
                                    if i+1 == d["group"]:
                                        newline += cuda_version
                                    else:
                                        newline += group
                        file.write(newline)
            except Exception as e:
                print("Exception whilst writing {:}\n > {:}".format(fpathe))
                return False

            return True

        else:
            print("Error: Example project file {:} does not exist.".format(example_vcxproj))
            return False
    else:
        print("Error: Example directory {:} does not exist.".format(example_dir))
        return False


def main():
    # Process command line arguments
    parser = argparse.ArgumentParser(
        description="Update the visual studio version of one or all flame gpu projects"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the new project, which should not already exist"
    )

    parser.add_argument(
        "version",
        type=str,
        help="New cuda version. I.e. 8.0, 9.0 or 9.1"
    )

    # parser.add_argument(
    #     "-f",
    #     "--force",
    #     action="store_true",
    #     help="Force destination directory to be replaced.",
    #     default=False
    # )

    args = parser.parse_args()


    # Get the location of this script, and calcualte relavant paths
    script_path = os.path.realpath(__file__)
    script_dir = os.path.split(script_path)[0]
    examples_dir = os.path.abspath(os.path.join(script_dir, RELATIVE_EXAMPLES_DIR))
    
    # Ensure the version is legit.
    if not checkVersion(args.version):
        print("Error: Version must be of the format {:}".format(VERSION_FORMAT))
        return False
    # Ensure the examples dir exists.
    if not examplesDirExists(examples_dir):
        print("Error: examples directory {:} does not exist".format(examples_dir))
        return False

    # Output a summary of what is attempted to be done.
    if args.name is not None:
        print("Updating CUDA version for {:} visual studio project to {:}".format(args.name, args.version))
    else:
        print("Updating CUDA version for all visual studio projects to {:}".format(args.version))


    if args.name is not None:
        return updateCudaVersionInVsproj(args.name, args.version, examples_dir)
    else:
        # Get all the example projects
        projects = findAllVcxprojFiles(examples_dir)
        statuses = []
        for project in projects:
            returncode = updateCudaVersionInVsproj(project, args.version, examples_dir)
            statuses.append(returncode)
        num_successes = statuses.count(True)
        print("{:} of {:} projects updated successfully.".format(num_successes, len(statuses)))
        return num_successes == len(statuses)


        

if __name__ =="__main__":
    success = main()
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)
