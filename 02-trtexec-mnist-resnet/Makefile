PYTHON = 3.9
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	conda install -y pytorch=1.10.2 torchvision=0.11.3 cudatoolkit=11.3 -c anaconda -c pytorch