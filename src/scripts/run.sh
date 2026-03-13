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

## TMPDIR=/mnt/disk3/yusheng/tmp_ncu XDG_RUNTIME_DIR=/mnt/disk3/yusheng/tmp_ncu ncu --set basic --csv --launch-skip 0 --launch-count 1 -- /mnt/disk3/yusheng/HeCBench/src/floydwarshall-cuda/main 1024 100 16 > metrics.csv 2> ncu.log

./run_ncu_bench.py --bench floydwarshall --bench-dir /mnt/disk3/yusheng/HeCBench/src \
    --ncu-binary ncu --ncu-set basic --launch-skip 0 --launch-count 1 \
    --ncu-out reports/ncu --metrics "sm_efficiency,achieved_occupancy" \
    --ncu-args ""

TMPDIR=/mnt/disk3/yusheng/tmp_ncu XDG_RUNTIME_DIR=/mnt/disk3/yusheng/tmp_ncu ./run_ncu_bench.py --bench cuda \
    --bench-dir /mnt/disk3/yusheng/HeCBench/src \
    --ncu-binary ncu --ncu-set basic \
    --launch-skip 0 --launch-count 100 \
    --ncu-out reports/ncu \
    --summary ncu_summary.csv \
    --keep-logs

TMPDIR=/mnt/disk3/yusheng/tmp_ncu XDG_RUNTIME_DIR=/mnt/disk3/yusheng/tmp_ncu ncu --set basic --csv --launch-skip 0 --launch-count 10 -- /mnt/disk3/yusheng/HeCBench/src/backprop-cuda/main 4096

python3 extract_reports.py --reports reports/ncu --set basic --out ncu_reports_summary.csv --metrics "Achieved Occupancy,Compute (SM) Throughput,DRAM Throughput,Memory Throughput"

