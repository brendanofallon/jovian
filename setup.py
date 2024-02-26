import os
import re
import subprocess

from setuptools import setup
from setuptools.command.install import install

requires=[]

packages = [
    "dnaseq2seq",
]

pip_requirements_file = "pip_requirements.txt"



def install_required_pip_packages():
    """
    This function installs external requirements to be installed through pip,
    defined in pip_requirements.txt
    :return:
    """
    if not os.path.exists(pip_requirements_file):
        return
    print(f"Installing packages defined in {pip_requirements_file}")
    cmd = [
        "python",
        "-m",
        "pip",
        "install",
        "-r",
        pip_requirements_file,
    ]
    print(f"Running command: {cmd}")
    subprocess.check_call(cmd)


class DNAseq2seqInstallCommand(install):
    """
    Customized setuptools install command
    """

    def run(self):
        install_required_pip_packages()
        install.run(self)


def parse_version():
    with open("dnaseq2seq/__init__.py", "r") as fd:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fd.read(), re.MULTILINE
        ).group(1)
    if not version:
        raise RuntimeError("Cannot find version information")
    return version


setup(
    name="dnaseq2seq",
    version=parse_version(),
    packages=packages,
    package_dir={"dnaseq2seq": "dnaseq2seq"},
    package_data={
        "dnaseq2seq": ["test/resources/*", ],
        "": ["*.yaml", "*.tsv", "*.txt"],
    },  # TODO add test documents with schema samples
    include_package_data=True,
    url="",
    license="",
    install_requires=requires,
    scripts=[
        "dnaseq2seq/main.py",
    ],
    cmdclass={"install": DNAseq2seqInstallCommand},
    tests_require=["pytest"],
    author="",
    author_email="",
    description="Variant caller using Transformers",
)

