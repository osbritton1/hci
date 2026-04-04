from skbuild import setup

setup(
    name="hci",
    version="0.1.0",
    description="An inefficient implementation of the HCI selection algorithm for PySCF",
    author="Simon Britton",
    license="Apache-2.0",
    packages=["hci"],
    python_requires=">=3.10",
)