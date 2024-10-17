from setuptools import setup, find_packages

setup(
    name="taskgen-alpha",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "openai>=1.3.6",
        "langchain",
        "dill>=0.3.7",
        "termcolor>=2.3.0",
        "requests",
        "PyPDF2",
        "python-docx",
        "pandas",
        "chromadb",
        "xlrd",
        "chromadb>=0.5.2",
        "asyncio"
    ],
)
