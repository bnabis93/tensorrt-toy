model-trt:
	mkdir $(CURDIR)/output
	docker run --gpus all -it \
		--user $(id -u tensorrt):$(id -g tensorrt) \
		-v $(CURDIR)/src:/workspace/src \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:22.03-py3 \
		/bin/bash -c "python src/onnx_exporter.py && \
			trtexec --onnx=./model.onnx \
					--saveEngine=./$(MODEL_PATH)/output/model.trt"

benchmark:
	docker run --gpus all -it \
		--user $(id -u tensorrt):$(id -g tensorrt) \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:22.03-py3 \
		/bin/bash -c "trtexec \
			--loadEngine=./$(MODEL_PATH)/output/model.trt \
			--batch=1"