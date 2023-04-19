cd multi_agent_path_planning/lifelong_MAPF
rm dataproducts/output.yaml -f > /dev/null 2>&1

# TASK_ALLOCATOR_CLASS_DICT for --allocator flags

python3 lifelong_MAPF.py dataproducts/input.yaml dataproducts/output.yaml --log INFO --random-seed 0 --allocator random_random && \
python3 visualize_lifelong.py dataproducts/input.yaml dataproducts/output.yaml