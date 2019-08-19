from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.readlines()

setup(
    name='midline_shift_detection',
    packages=find_packages(include=('midline_shift_detection',)),
    url='https://github.com/neuro-ml/midline-shift-detection',
    # install_requires=requirements, TODO: uncomment after next release of dpipe
)
