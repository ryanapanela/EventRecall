from setuptools import setup, find_packages

setup(
    name="event_recall_tool",
    version="0.1.0",
    description="Automated Event Segmentation and Recall Scoring Tool",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "openai",
        "backoff",
        "transformers",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "event-recall-tool=event_recall_tool.cli:main"
        ]
    },
    python_requires=">=3.8",
)
