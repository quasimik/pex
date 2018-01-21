# Pex

## Installation (Ubuntu 16.04)

### Flask
1. Install Python 3 and pip `sudo apt-get install python3 python-pip`
2. Install virtualenv using pip `sudo pip install virtualenv`
3. Make a virtualenv instance venv in project root `virtualenv venv`
4. Activate the instance `. venv/bin/activate`
5. Install Flask (in venv) `pip install Flask`
6. Export the FLASK_APP environment variable `export FLASK_APP=server.py`
7. Run Flask in venv `flask run`

### OpenPose
1. Follow the install instructions for [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/), making sure to install the libraries to local path
2. Run `cmake .` to generate Makefiles using CMake.
3. `cd models` and run `./getModels.sh` to download the OpenPose models.
4. Compile in the main folder using `make` to get `poser.bin`.
