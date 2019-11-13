
deps: deps/apex deps/warp-transducer

deps/apex:
	git clone https://github.com/NVIDIA/apex deps/apex && \
	cd deps/apex && \
	git checkout 880ab925bce9f817a93988b021e12db5f67f7787 && \
	pip install -v --no-cache-dir --global-option="--cpp_ext" \
			--global-option="--cuda_ext" ./ && \
	cd ..

deps/warp-transducer:
	git clone https://github.com/HawkAaron/warp-transducer deps/warp-transducer && \
	cd deps/warp-transducer && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make VERBOSE=1 && \
	export WARP_RNNT_PATH=`pwd` && \
	export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME && \
	export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH" && \
	export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH && \
	export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
	export CFLAGS="-I$CUDA_HOME/include $CFLAGS" && \
	cd ../pytorch_binding && \
	python3 setup.py install --user && \
	rm -rf ../tests test ../tensorflow_binding && \
	cd ../../..
