# viabill_case
*NOTE:* Assuming the data seems to be synthetic some modelling activities were simplified, prioritizing code modularity, testing and experiment tracking. The goal of the project was to show the overall coding style, approach to code testing (only tests for data related code were created assuming relatively limited time), environment and repository management, usage of tools for hyperparameters optimization and experiment tracking. Additionally simple web-app may be started on the localhost by running Docker container. Some possible improvements in terms of modelling may be the following:
- Analyze predictions and optimize decision threshold for defaults. In other words the default threshold of 50% may be not optimal and we may tweak it to achive better metrics;
- Deeper multivariate analysis to find some additional hidden dependancies among independant variables; 
- Try other model types;
- Broader feature engineering.

## Table of contents
- [Pre-requirements](#pre-requirements)
- [Installation](#installation)
- [Usage](#usage)

## Pre-requirements
You need [docker](https://www.docker.com/) to be able to run containers with web application. Additionally it's preferable to have [make](https://www.gnu.org/software/make/) for easy project installation.

## Installation
Clone the repo, create `data` folder and locate the data (`viabill.db`) to it.

There are two ways to install project for usage.

1. The easiest way is to use `make` and simply run the command:
```
make setup
```

2. If you prefer another way to manage virtual environment or don't have `make` you can always use `requirements.txt` file to build all the dependancies for your own environment.

If you want to run `Report.ipynb` notebook, make sure you use respective kernel. 

## Usage

To run the experiments with `MLFlow` and `hyperopt` after the environment is configured run command from root directory in your CLI.

```
python run_experiments.py
```

Then, you can promote best model to Production stage and prepare inference folder for containarization with Docker.
```
python promote_last_best.py
```

Afterwards, you can go to `inference/` folder and build Docker container.
```
docker build -t default-prediction-service:v1 .
```

Run your container and set web application in active state
```
docker run -it --rm -p 9696:9696  default-prediction-service:v1
```

Now you can send requests to web-service under `http://localhost:9696/app` URL. 

