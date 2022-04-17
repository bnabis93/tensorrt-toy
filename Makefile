PYTHON = 3.9
BASENAME=$(shell basename $(CURDIR))
CONDA_CH=conda-forge defaults pytorch

env:
	conda create -n $(BASENAME)  python=$(PYTHON)
setup:
	conda install --file requirements.txt $(addprefix -c ,$(CONDA_CH))
	
train:
	python src/train.py

format:
	black src .
	isort src .

lint:
	pytest src . --flake8 --pylint --mypy