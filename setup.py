from setuptools import find_packages, setup

setup(
    name='structiou',
    version='0.1',
    description='Structured intersection over union ratio of two (speech) constituency parse trees',
    author='Freda Shi',
    packages=find_packages(exclude=['tests']),
    setup_requires=[
        'setuptools',
    ],
    install_requires=[
        'numpy>=1.21.6',
        'nltk',
    ],
    python_requires='>=3.7',
)
