from setuptools import setup, find_packages

setup(
    name='quantumsimulationlib',
    version='0.0.11',
    author='Vytis Krupovnickas',
    author_email='vytis000@gmail.com',
    description='A simple quantum computing simulator focusing on quantum walks and machine learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Vytis-K/QuantumSimulationLib',
    packages=find_packages(where='src'),  # Correctly use this single declaration
    package_dir={'': 'src'},  # Points setuptools to the src directory
    install_requires=[
        'numpy',
        'matplotlib',
        'qiskit',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)
