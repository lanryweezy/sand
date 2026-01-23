from setuptools import setup, find_packages

setup(
    name="silicon-intelligence",
    version="0.1.0",
    description="AI-powered physical implementation system for IC design",
    author="Silicon Intelligence Team",
    author_email="team@silicon-intelligence.ai",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "networkx>=2.8.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "pyverilog>=1.3.0",
        "lark-parser>=0.12.0",
        "antlr4-python3-runtime>=4.11.0",
        "gdstk>=0.9.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "silicon-intel=silicon_intelligence.main:main"
        ]
    },
    python_requires=">=3.8",
)