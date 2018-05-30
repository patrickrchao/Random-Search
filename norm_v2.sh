#/usr/local/bin/python3
# Arbitrary Condition
python3 RandomSearchV2.py
python3 RandomSearchV2.py --alpha 0.03
python3 RandomSearchV2.py --alpha 0.003
python3 RandomSearchV2.py --nu 0.03
python3 RandomSearchV2.py --nu 0.003

python3 RandomSearchV2.py --mat_size 10
python3 RandomSearchV2.py --alpha 0.03 --mat_size 10
python3 RandomSearchV2.py --alpha 0.003 --mat_size 10
python3 RandomSearchV2.py --nu 0.03 --mat_size 10
python3 RandomSearchV2.py --nu 0.003 --mat_size 10

# Condition <10 for mat_size 5 and <100 for mat_size 10
python3 RandomSearchV2.py --max_condition 10 
python3 RandomSearchV2.py --alpha 0.03 --max_condition 10
python3 RandomSearchV2.py --alpha 0.003 --max_condition 10
python3 RandomSearchV2.py --nu 0.03 --max_condition 10
python3 RandomSearchV2.py --nu 0.003 --max_condition 10

python3 RandomSearchV2.py --mat_size 10 --max_condition 100
python3 RandomSearchV2.py --alpha 0.03 --mat_size 10 --max_condition 100
python3 RandomSearchV2.py --alpha 0.003 --mat_size 10 --max_condition 100
python3 RandomSearchV2.py --nu 0.03 --mat_size 10 --max_condition 100
python3 RandomSearchV2.py --nu 0.003 --mat_size 10 --max_condition 100

# Condition between 10 and 100 for mat_size 5, 100 and 1000 for mat_size 10
python3 RandomSearchV2.py --min_condition 10 --max_condition 100
python3 RandomSearchV2.py --alpha 0.03  --min_condition 10 --max_condition 100
python3 RandomSearchV2.py --alpha 0.003  --min_condition 10 --max_condition 100
python3 RandomSearchV2.py --nu 0.03  --min_condition 10 --max_condition 100
python3 RandomSearchV2.py --nu 0.003  --min_condition 10 --max_condition 100

python3 RandomSearchV2.py --mat_size 10  --min_condition 100 --max_condition 1000
python3 RandomSearchV2.py --alpha 0.03 --mat_size 10  --min_condition 100 --max_condition 1000
python3 RandomSearchV2.py --alpha 0.003 --mat_size 10  --min_condition 100 --max_condition 1000
python3 RandomSearchV2.py --nu 0.03 --mat_size 10  --min_condition 100 --max_condition 1000
python3 RandomSearchV2.py --nu 0.003 --mat_size 10  --min_condition 100 --max_condition 1000

# Condition >100 for mat_size 5, >1000 for mat_size 10
python3 RandomSearchV2.py --min_condition 100
python3 RandomSearchV2.py --alpha 0.03 --min_condition 100
python3 RandomSearchV2.py --alpha 0.003 --min_condition 100
python3 RandomSearchV2.py --nu 0.03 --min_condition 100
python3 RandomSearchV2.py --nu 0.003 --min_condition 100

python3 RandomSearchV2.py --mat_size 10 --min_condition 1000
python3 RandomSearchV2.py --alpha 0.03 --mat_size 10 --min_condition 1000
python3 RandomSearchV2.py --alpha 0.003 --mat_size 10 --min_condition 1000
python3 RandomSearchV2.py --nu 0.03 --mat_size 10 --min_condition 1000
python3 RandomSearchV2.py --nu 0.003 --mat_size 10 --min_condition 1000
