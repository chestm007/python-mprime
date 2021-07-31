from distutils.core import setup

with open("README.md") as f:
    readme = f.read()


setup(
    name="mprime",
    version="PROJECTVERSION",
    packages=["mprime"],
    url="https://github.com/chestm007/python-mprime",
    license="GPL-2.0",
    author="Max Chesterfield",
    author_email="chestm007@hotmail.com",
    maintainer="Max Chesterfield",
    maintainer_email="chestm007@hotmail.com",
    description="mprime wrapper",
    long_description=readme,
    install_requires=[],
    entry_points="""
        [console_scripts]
        python-mprime=mprime.mprime:main
    """,
)
