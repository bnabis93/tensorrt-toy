convert-torch2trt:
	mkdir -p $(CURDIR)/output
	docker run --gpus 1 -it \
		-v $(CURDIR)/src:/workspace/src \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:22.03-py3 \
		/bin/bash -c "python src/onnx_exporter.py && \
			trtexec --onnx=./model.onnx \
					--saveEngine=./output/model.trt"

benchmark:
	docker run --gpus 1 -it \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:22.03-py3 \
		/bin/bash -c "trtexec \
			--loadEngine=./output/model.trt \
			--batch=1"