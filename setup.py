import os
import re
import subprocess

from setuptools import setup
from setuptools.command.install import install

requires=[]

packages = [
    "dnaseq2seq",
]

class DNAseq2seqInstallCommand(install):
    """
    Customized setuptools install command
    """

    def run(self):
        #if "DOCKER_VERSION" not in os.environ:
        #    install_required_conda_packages()
        #    install_required_pip_packages()
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
        "dnaseq2seq/bin/main.py",
    ],
    cmdclass={"install": DNAseq2seqInstallCommand},
    tests_require=["pytest"],
    author="",
    author_email="",
    description="Variant caller using Transformers",
)
