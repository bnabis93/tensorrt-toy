PYTHON = 3.9
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch

train:
	python src/train.py

convert-trt:
	