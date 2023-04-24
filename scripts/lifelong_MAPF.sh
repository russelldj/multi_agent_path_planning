cd multi_agent_path_planning/lifelong_MAPF
rm dataproducts/output.yaml -f > /dev/null 2>&1

# See TASK_ALLOCATOR_CLASS_DICT for --allocator flags
# Options formatted as (task allocator)_(idle goal generator)
# random_random
# random_kmeans
# random_current
# linear_sum_random
# linear_sum_kmeans
# linear_sum_current

python3 lifelong_MAPF.py dataproducts/input.yaml dataproducts/output.yaml --log INFO --random-seed 0 --allocator random_random && \
python3 visualize_lifelong.py dataproducts/input.yaml dataproducts/output.yaml