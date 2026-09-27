from setuptools import setup, find_packages

setup(
    name="flightning",
    version="0.1.0",
    description="Learning visuomotor quadrotor flight through dynamic narrow gaps "
                "using differentiable simulation.",
    author="Riccardo Bianco",
    author_email="rbianco@student.ethz.ch",
    maintainer="Riccardo Bianco",
    maintainer_email="rbianco@student.ethz.ch",
    # Derived from rpg_flightning by Johannes Heeg et al. (UZH-RPG), GPL-3.0.
    license="GPL-3.0",
    packages=find_packages(),
    install_requires=[
        # add your project dependencies here
    ],
)
