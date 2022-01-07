from setuptools import setup

setup(
    name             = "torchsummaryDynamic",
    version          = "1.0.0",
    description      = "Improved real/dynamic FLOPs calculation tool of torchsummaryX.",
    author           = "chenbong",
    author_email     = "bhchen@stu.xmu.edu.cn",
    url              = "https://github.com/chenbong/torchsummaryDynamic",
    packages         =["torchsummaryDynamic"],
    install_requires = ["torch", "numpy", "pandas"],
)