import io
import os

from setuptools import setup, find_packages

SETUP_DIR = os.path.dirname(os.path.abspath(__file__))


# List all of your Python package dependencies in the
# requirements.txt file

def readfile(filename, split=False):
    with io.open(filename, encoding="utf-8") as stream:
        if split:
            return stream.read().split("\n")
        return stream.read()


readme = readfile("README.rst", True)
# For requirements not hosted on PyPi place listings
# into the 'requirements.txt' file.
requires = [
    # minimal requirements listing
    "opencmiss.utils >= 0.3",
    "opencmiss.zinc >= 3.7"
]
readme.extend(['', 'License', '=======', '', '::', ''])
source_license = readfile("LICENSE")

setup(
    name="fieldfitter",
    version="0.2.0",
    description="Scaffold/model field fitting library using OpenCMISS-Zinc.",
    long_description="\n".join(readme) + source_license,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    author="Auckland Bioengineering Institute",
    author_email="r.christie@auckland.ac.nz",
    url="https://github.com/ABI-Software/fieldfitter",
    license="Apache Software License",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    install_requires=requires,
)
