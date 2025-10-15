from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="EventRecall",
    version="0.1.0",
    description="Event Segmentation Applications for LLM-Enabled Automated Recall Scoring",
    author="Ryan A. Panela",
    py_modules=["segmentation", "recall", "cli", "utils"],
    package_dir={"": "module"},
    #packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)
