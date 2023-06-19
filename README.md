# nerf-papers
Implementation of papers from scratch  in the field of Neural radiance field to build my own understanding.

This is also my follow along work for the course given Maxime Vandegar on Udemy. <br>
Big thanks to him for helping me build understanding on a lot of concepts. <br>
Some papers were mentioned in course, others I have implemented. 
Implementation may be different from original paper to run on my 2080 GPU.

## Installation

1. This repository uses `Python 3.8`. Install it.
2. Install `poetry`. <br>
This repository uses `poetry` for dependency management. Checkout the `pyproject.toml` to see the details of the dependencies. <br>
You can install poetry from [official site](https://python-poetry.org/docs/).
2. Once `poetry` is installed, install the dependencies:
```commandline
poetry install
```
Be patient, this step may take some time.


## Notebooks

The notebooks are in `notebooks` directory.
Order of review:
1. `3Dreconstruction`
2. `voxel_reconstruction`
3. `nerf_reconstruction`
4. `fourier_nerf`
5. `ingp-1`


## Acknowledgement

Course link: [Neural Radiance Fields](https://www.udemy.com/course/neural-radiance-fields-nerf/) <br>
Check out: [YT](https://www.youtube.com/@papersin100linesofcode) 