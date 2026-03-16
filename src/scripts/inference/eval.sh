cp bpnn_layerforward_r1.h testcases/backprop-cuda/bpnn_layerforward.h
cd testcases/backprop-cuda
make clean
make
./main 4096

# ncu
cp bpnn_layerforward_r1.h testcases/backprop-cuda/bpnn_layerforward.h
cd testcases/backprop-cuda
make clean
make
ncu ./main 4096

##
sudo ncu ./main 4096
