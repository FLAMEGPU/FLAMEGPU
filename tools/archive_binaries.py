"""
Script to extract the required files for running standalone. 

@todo - make this executable from any location (use relative to .py paths)
"""
import os 
import glob
import shutil
import argparse
import errno
import zipfile



DEFAULT_OUTPUT = "example_binaries.zip"

PATTERNS = [
    '../bin/x64/*.bat',
    '../bin/x64/Release_*/*.exe',
    '../bin/x64/Release_*/*.dll',
    '../examples/*/iterations/*.xml',
    '../media/*'
]

def run(args):
    script_loc = os.path.dirname(os.path.realpath(__file__))
    # Create the archive
    zipf = zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED)

    os.chdir(script_loc)
    # Add each file to zip
    for pattern in PATTERNS:
        pattern = os.path.normpath(os.path.join(script_loc, pattern))
        print(pattern)
        for file in glob.glob(pattern):
            # Normalise the file path
            input_loc = os.path.normpath(file)
            output_loc = os.path.relpath(input_loc, '../')
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
