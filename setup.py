from setuptools import setup


setup(
    name="veritas",
    version="0.9.2",
    description="Machine Learning part of Veritas.",
    author="Jianfei Gao",
    author_email="gao462@purdue.edu",
    packages=["veritas"],
    package_dir={"veritas": "./src/veritas"},
    license="MIT License",
    zip_safe=False,
)
