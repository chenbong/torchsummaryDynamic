from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name             = "torchsummaryDynamic",
    version          = "0.0.3",
    description      = "Improved real/dynamic FLOPs calculation tool of torchsummaryX.",
    author           = "chenbong",
    author_email     = "bhchen@stu.xmu.edu.cn",
    url              = "https://github.com/chenbong/torchsummaryDynamic",
    packages         =["torchsummaryDynamic"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = ["torch", "numpy", "pandas"],
)