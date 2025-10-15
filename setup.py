from setuptools import setup, find_packages

setup(
    name="EventRecall",
    version="0.1.0",
    description="Event Segmentation Applications for LLM-Enabled Automated Recall Scoring",
    author="Ryan A. Panela",
    py_modules=["segmentation", "recall", "cli", "utils"],
    package_dir={"": "module"},
    #packages=find_packages(),
    install_requires=[
        "openai",
        "backoff",
        "transformers",
        "tqdm"
    ],
    python_requires=">=3.8",
)
