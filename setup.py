"""
Setup script for Movie Recommendation ETL Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="movie-recommendation-etl",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready ETL pipeline for movie recommendation system with sub-second response times",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/movie-recommendation-etl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Database",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "performance": [
            "numba>=0.56.0",
            "joblib>=1.1.0",
            "cython>=0.29.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "movie-etl=main_etl_pipeline:main",
            "movie-recommend=recommend:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "etl", "pipeline", "movie", "recommendation", "machine-learning", 
        "collaborative-filtering", "content-based", "data-processing",
        "sqlite", "pandas", "scikit-learn"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/movie-recommendation-etl/issues",
        "Source": "https://github.com/your-username/movie-recommendation-etl",
        "Documentation": "https://movie-recommendation-etl.readthedocs.io/",
    },
)