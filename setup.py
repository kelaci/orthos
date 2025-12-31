from setuptools import setup, find_packages

setup(
    name="gaia-neuro",
    version="4.1.0",
    description="Generalized Adaptive Intelligent Architecture - Biologically plausible neural networks",
    author="GAIA Team",
    author_email="dev@gaia.ai",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "gpu": ["cupy>=9.0.0"],
        "dev": ["pytest", "black", "mypy", "flake8"]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
