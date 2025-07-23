from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-field-technician",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered field technician assistant for equipment diagnosis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-field-technician",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "openai>=1.3.0", 
        "langchain>=0.0.340",
        "faiss-cpu>=1.7.4",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "plotly>=5.17.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "field-technician=field_technician_ai:main",
        ],
    },
)