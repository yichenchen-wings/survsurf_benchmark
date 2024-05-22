#!/bin/bash
pwd
ENV_NAME=$1
eval "$(conda shell.bash hook)"

mamba env create -n $ENV_NAME -f environment.yml --force
conda activate $ENV_NAME
PIP_PATH_IN_USE=$(which pip)

echo "pip in use: $PIP_PATH_IN_USE"
if [[ "$PIP_PATH_IN_USE" == *"$ENV_NAME"* ]]
then
  pip install -r ./pip_requirements.txt
  echo "pip install has been run successfully"
else
  echo "pip in use does not come from the specified conda environment, pip install aborted"
fi

