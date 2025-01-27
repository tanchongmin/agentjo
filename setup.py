from setuptools import setup, find_packages

setup(
    name="agentjo",
    version="0.0.5",
    packages=find_packages(),
    install_requires=[
        "openai>=1.3.6",
        "dill>=0.3.7",
        "termcolor>=2.3.0",
        "requests",
        "pandas",
    ],
    extras_require={
        "full": [
            "langchain",
            "PyPDF2",
            "python-docx",
            "xlrd",
            "sentence_transformers",
        ],
    },
)
