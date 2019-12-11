
deps: deps/apex deps/warp-transducer

deps/apex:
	git clone https://github.com/NVIDIA/apex deps/apex && \
	cd deps/apex && \
	git checkout 880ab925bce9f817a93988b021e12db5f67f7787 && \
	pip install -v --no-cache-dir --global-option="--cpp_ext" \
			--global-option="--cuda_ext" ./ && \
	cd .. && rm -rf apex

deps/warp-transducer:
	export CFLAGS="-I$(CUDA_HOME)/include $(CFLAGS)" && \
	git clone https://github.com/HawkAaron/warp-transducer deps/warp-transducer && \
	cd deps/warp-transducer && \
	git checkout c6d12f9e1562833c2b4e7ad84cb22aa4ba31d18c && \
	mkdir build && \
	cd build && \
	export WARP_RNNT_PATH=$(CONDA_PREFIX)/lib && \
	cmake .. && \
	make && \
	mv libwarprnnt.so  $(CONDA_PREFIX)/lib && \
	cd ../pytorch_binding && \
	python3 setup.py install --user && \
	cd ../../../ && \
  rm -rf deps && \
  echo "export WARP_RNNT_PATH=$(CONDA_PREFIX)/lib" >> ~/.bashrc
