## Development
### Setup local environment
1. Install Poetry https://python-poetry.org/docs/#installation
2. (Window) If there are multiple version of python, rename make a copy of C:\Program Files\Python37\python.exe into python3.7.exe
3. Run `poetry install` under project folder with pyproject.toml, and virtual environment will be created under `<project_folder>/.venv`
4. Add virtual environment to Pycharm
5. Add new python package with `poetry add <package_name>`


### New project
1. Run `poetry new <project_name>` under the `PycharmProjects` folder
1. Before running `poetry install`
    * run `poetry config virtualenvs.in-project true --local`
    * or make sure run `virtualenvs.in-project true` is set in poetry.toml


### Poetry version specification
Python versioning: major.minor.patch
* ^: same major, newer minor
    * beepboop = "^2.1.7" # Equivalent to >=2.1.7, <3.0.0
* ~: newer patch unless only major is specified
    * beepboop = "~2.1.7" # Equivalent to >=2.1.7, <2.2.0
    * beepboop = "~2.1" # Equivalent to >=2.1.0, <2.2.0
    * beepboop = "~2" # Equivalent to >=2.0.0, <3.0.0
    
https://stackoverflow.com/questions/54720072/dependency-version-syntax-for-python-poetry
