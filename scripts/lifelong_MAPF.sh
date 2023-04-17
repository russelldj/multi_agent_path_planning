cd multi_agent_path_planning/lifelong_MAPF
rm dataproducts/output.yaml -f > /dev/null 2>&1
python3 lifelong_MAPF.py dataproducts/input.yaml dataproducts/output.yaml --log INFO --random-seed 0 --allocator linear_sum && \
python3 visualize_lifelong.py dataproducts/input.yaml dataproducts/output.yaml