PYTHON = 3.9
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch

train:
	python src/train.py

convert-trt:
	mkdir -p $(CURDIR)/output
	docker run --gpus 1 -it \
		-v $(CURDIR)/src:/workspace/src \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:22.03-py3 \
		/bin/bash -c "python src/onnx_exporter.py && \
			trtexec --onnx=./model.onnx \
					--saveEngine=./output/model.trt"