from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='voxelmon',
    version='0.0.2',
    author='Johnathan Tenny',
    author_email='jt893@nau.edu',
    description='',
    #long_description=open('README.md').read(),
    #long_description_content_type='text/markdown',
    url='https://github.com/j-tenny/voxelmon',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ]
)