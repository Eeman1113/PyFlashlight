import setuptools
from setuptools.command.install import install
import subprocess

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

apt_dependencies = [
    'nvidia-cuda-toolkit'
]

for package in apt_dependencies:
    subprocess.check_call(['sudo', 'apt', 'install', package])

class CustomInstall(install):
    def run(self):
        subprocess.call(['make', '-C', 'build'])
        install.run(self)


setuptools.setup(
    name = "pyflashlight",
    version = "0.0.3",
    author = "Lucas de Lima",
    author_email = "nogueiralucasdelima@gmail.com",
    description = "A deep learning framework",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/lucasdelimanogueira/pyflashlight",
    project_urls = {
        "Bug Tracker": "https://github.com/lucasdelimanogueira/pyflashlight/issues",
        "Repository": "https://github.com/lucasdelimanogueira/pyflashlight"
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages = setuptools.find_packages(),
    package_data={'pyflashlight': ['csrc/*', 'libtensor.so']},
    cmdclass={
        'install': CustomInstall,
    },
    python_requires = ">=3.6"
)
