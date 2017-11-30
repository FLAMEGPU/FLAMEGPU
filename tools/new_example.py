#! /bin/python

"""
Simple script to create a new FLAME GPU example, based on an existing example.

@author Peter Heywood <p.heywood@sheffield.ac.uk>
"""


import argparse
import sys
import os
import shutil
import distutils.dir_util

RELATIVE_EXAMPLES_DIR="../examples"

DEFAULT_BASE="EmptyExample"

FILES_TO_CHANGE = ["Makefile", "{:}.sln", "{:}.vcxproj"]

def examplesDirExists(examples_dir):
    # Check if the examples directory exists.
    return os.path.isdir(examples_dir)

def targetExists(target_dir):
    return os.path.exists(target_dir)

def baseExists(base_dir):
    return os.path.exists(base_dir)

def rename_files(target_dir, target_name, base_name):
    try: 
        for subdir, dirs, files in os.walk(target_dir):
            for file in files:
                # If the file starts with the bsae_name, replace it.
                if file.startswith(base_name):
                    # @todo - only rename certain files.
                    new_file = file.replace(base_name, target_name)
                    shutil.move(os.path.join(subdir, file), os.path.join(subdir, new_file))
    except Exception as e:
        print("Exception renaming files:\n > {:}".format(fpathe))
        return False
    return True

def update_files(target_dir, target_name, base_name):
    # For each file we wish to change
    for file_pattern in FILES_TO_CHANGE:
        # Find the filepath
        fname = file_pattern.replace("{:}", target_name)
        fpath = os.path.join(target_dir, fname)
        # If the file exists.
        if os.path.isfile(fpath):
            # Open the path for read 
            content = None
            try:
                with open(fpath, "r") as file:
                    content = file.readlines()
            except Exception as e: 
                print("Exception whilst reading {:}\n > {:}".format(fpath, e))
                return False

            try:
                with open(fpath, "w") as file:
                    for line in content:
                        newline = line.replace(base_name, target_name)
                        file.write(newline)
            except Exception as e:
                print("Exception whilst writing {:}\n > {:}".format(fpathe))
                return False
    return True

def createExample(target_name, target_dir, base_name, base_dir):
    try:
        # Copy the base directory to the target directory
        distutils.dir_util.copy_tree(base_dir, target_dir)

        # Assuming this copy completes successfully, any files which start with the base name must be replaced. 
        renamed = rename_files(target_dir, target_name, base_name)
        if renamed:
            # Update the contents of key files.
            updated = update_files(target_dir, target_name, base_name)
            return updated
        else :
            return False
    except Exception as e:
        print("An Exception has occurred:\n  {:}".format(e) )
        return False


def main():
    # Process command line arguments
    parser = argparse.ArgumentParser(
        description="Create and rename a copy of a FLAME GPU example, as a starting point for new projects"
    )
    parser.add_argument(
        "name",
        type=str,
        help="Name of the new project, which should not already exist"
    )

    parser.add_argument(
        "--base",
        type=str,
        help="The name of the existing example project to be based on",
        default=DEFAULT_BASE
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force destination directory to be replaced.",
        default=False
    )

    args = parser.parse_args()


    # Get the location of this script, and calcualte relavant paths
    script_path = os.path.realpath(__file__)
    script_dir = os.path.split(script_path)[0]
    examples_dir = os.path.abspath(os.path.join(script_dir, RELATIVE_EXAMPLES_DIR))

    # Output a summary of what is attempted to be done.
    print("Creating new example `{:}` based on `{:}` in `{:}` ".format(args.name, args.base, examples_dir))

    # Ensure the examples dir exists.
    if not examplesDirExists(examples_dir):
        print("Error: examples directory {:} does not exist".format(examples_dir))
        return False

    # Construct some paths.
    base_dir = os.path.abspath(os.path.join(examples_dir, args.base))
    target_dir = os.path.abspath(os.path.join(examples_dir, args.name))

    # If the base does not exist, abort.
    if not baseExists(base_dir):
        print("Error: base model {:} does not exist".format(base_dir))
        return False

    # Check if the target project already exists, abd we are not forcing replacement.
    if targetExists(target_dir) and not args.force:
        # If it exists, abort.
        print("Error: target example directory {:} already exists. Use `--force` to overwrite.".format(target_dir))
        return False

    # Otherwise, unless we have any race conditions at this point, we can try to proceed.

    try:
        created = createExample(args.name, target_dir, args.base, base_dir)
        if created:
            print("Example `{:}` successfully created at `{:}`".format(args.name, target_dir))
            return True
        else:
            print("Error: Could not create the example. Please try again.")
            return False
    except Exception as e:
        print("Error: an exception occurred.\n > {:}".format(e))
        return False

        

if __name__ =="__main__":
    success = main()
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)
