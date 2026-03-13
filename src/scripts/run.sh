./autohecbench.py backprop-cuda --yes-prompt -o cuda.csv --verbose
./autohecbench.py backprop-sycl --sycl-type cpu --compiler-name icpx --yes-prompt -o cpu.csv --verbose

./autohecbench-compare.py cpu.csv cuda.csv

# ./autohecbench.py backprop-cuda --verbose

./autohecbench.py cuda -o cuda.csv --yes-prompt
./autohecbench.py sycl --sycl-type cpu --compiler-name icpx -o cpu.csv --yes-prompt
./autohecbench-compare.py cpu.csv cuda.csv

./autohecbench.py cuda -o cuda.csv --yes-prompt # -r 1
./autohecbench.py sycl --sycl-type cpu --compiler-name icpx -o cpu.csv --yes-prompt # -r 1
./autohecbench-compare.py cpu.csv cuda.csv -o comparison.csv