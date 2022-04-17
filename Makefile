<<<<<<< HEAD
PYTHON = 3.9
BASENAME=$(shell basename $(CURDIR))
CONDA_CH=conda-forge defaults pytorch

env:
	conda create -n $(BASENAME)  python=$(PYTHON)
setup:
	conda install --file requirements.txt $(addprefix -c ,$(CONDA_CH))
train:
	python src/train.py


=======
docker:
	docker run --gpus all -it --rm -v$(CURDIR)/Makefile:/workspace/Makefile -v $(CURDIR)/src:/workspace/src nvcr.io/nvidia/pytorch:22.03-py3

docker-cpu:
	docker run -it --rm -v$(CURDIR)/Makefile:/workspace/Makefile -v $(CURDIR)/src:/workspace/src nvcr.io/nvidia/pytorch:22.03-py3

torch2trt:
	git clone https://github.com/NVIDIA-AI-IOT/torch2trt
	cd torch2trt && python3 setup.py install
>>>>>>> 42c6fef8eb9cf10cf9aa42061ee9c8ea9d0a86d1

format:
	black src .
	isort src .

lint:
	pytest src . --flake8 --pylint --mypy