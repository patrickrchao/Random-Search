#/usr/local/bin/python3
# Arbitrary Condition
python3 RandomSearchV2.py --sigma_plots
python3 RandomSearchV2.py --alpha 0.03 --sigma_plots
python3 RandomSearchV2.py --alpha 0.003 --sigma_plots
python3 RandomSearchV2.py --nu 0.03 --sigma_plots
python3 RandomSearchV2.py --nu 0.003 --sigma_plots

python3 RandomSearchV2.py --mat_size 10 --sigma_plots
python3 RandomSearchV2.py --alpha 0.03 --mat_size 10 --sigma_plots
python3 RandomSearchV2.py --alpha 0.003 --mat_size 10 --sigma_plots
python3 RandomSearchV2.py --nu 0.03 --mat_size 10 --sigma_plots
python3 RandomSearchV2.py --nu 0.003 --mat_size 10 --sigma_plots

# Condition <10 for mat_size 5 and <100 for mat_size 10
python3 RandomSearchV2.py --sigma_plots --max_condition 10
python3 RandomSearchV2.py --alpha 0.03 --sigma_plots --max_condition 10
python3 RandomSearchV2.py --alpha 0.003 --sigma_plots --max_condition 10
python3 RandomSearchV2.py --nu 0.03 --sigma_plots --max_condition 10
python3 RandomSearchV2.py --nu 0.003 --sigma_plots --max_condition 10

python3 RandomSearchV2.py --mat_size 10 --sigma_plots --max_condition 100 
python3 RandomSearchV2.py --alpha 0.03 --mat_size 10 --sigma_plots --max_condition 100 
python3 RandomSearchV2.py --alpha 0.003 --mat_size 10 --sigma_plots --max_condition 100
python3 RandomSearchV2.py --nu 0.03 --mat_size 10 --sigma_plots --max_condition 100
python3 RandomSearchV2.py --nu 0.003 --mat_size 10 --sigma_plots --max_condition 100

# Condition between 10 and 100 for mat_size 5, 100 and 1000 for mat_size 10
python3 RandomSearchV2.py --sigma_plots --min_condition 10 --max_condition 100
python3 RandomSearchV2.py --alpha 0.03 --sigma_plots --min_condition 10 --max_condition 100
python3 RandomSearchV2.py --alpha 0.003 --sigma_plots --min_condition 10 --max_condition 100
python3 RandomSearchV2.py --nu 0.03 --sigma_plots --min_condition 10 --max_condition 100
python3 RandomSearchV2.py --nu 0.003 --sigma_plots --min_condition 10 --max_condition 100

python3 RandomSearchV2.py --mat_size 10 --sigma_plots --min_condition 100 --max_condition 1000
python3 RandomSearchV2.py --alpha 0.03 --mat_size 10 --sigma_plots --min_condition 100 --max_condition 1000
python3 RandomSearchV2.py --alpha 0.003 --mat_size 10 --sigma_plots --min_condition 100 --max_condition 1000
python3 RandomSearchV2.py --nu 0.03 --mat_size 10 --sigma_plots --min_condition 100 --max_condition 1000
python3 RandomSearchV2.py --nu 0.003 --mat_size 10 --sigma_plots --min_condition 100 --max_condition 1000

# Condition >100 for mat_size 5, >1000 for mat_size 10
python3 RandomSearchV2.py --sigma_plots --min_condition 100
python3 RandomSearchV2.py --alpha 0.03 --sigma_plots --min_condition 100
python3 RandomSearchV2.py --alpha 0.003 --sigma_plots --min_condition 100
python3 RandomSearchV2.py --nu 0.03 --sigma_plots --min_condition 100
python3 RandomSearchV2.py --nu 0.003 --sigma_plots --min_condition 100

python3 RandomSearchV2.py --mat_size 10 --sigma_plots --min_condition 1000
python3 RandomSearchV2.py --alpha 0.03 --mat_size 10 --sigma_plots --min_condition 1000
python3 RandomSearchV2.py --alpha 0.003 --mat_size 10 --sigma_plots --min_condition 1000
python3 RandomSearchV2.py --nu 0.03 --mat_size 10 --sigma_plots --min_condition 1000
python3 RandomSearchV2.py --nu 0.003 --mat_size 10 --sigma_plots --min_condition 1000
