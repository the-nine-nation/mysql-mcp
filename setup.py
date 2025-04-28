from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mysql-readonly-mcp",
    version="0.1.0",
    author="Lucas Luo",
    author_email="",
    description="MySQL Readonly MCP Server for accessing MySQL databases through MCP protocol",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/mysql-readonly-mcp",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "aiomysql",
        "mcp",
    ],
    entry_points={
        "console_scripts": [
            "mysql-mcp=src.mysql_mcp:main",
        ],
    },
) 