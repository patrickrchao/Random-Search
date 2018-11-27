#/usr/local/bin/python3
python3 main.py --function QUADRATIC  --step_function CONSTANT --function_param 2 --condition_num 1 --optimal --surface --contour
python3 main.py --function LOG --step_function CONSTANT --function_param 0.1 --condition_num 1 --optimal --surface --contour

python3 main.py --function QUADRATIC  --step_function CONSTANT --function_param 0.2 --condition_num 5 --optimal --surface --contour
python3 main.py --function LOG --step_function CONSTANT --function_param 0.0001 --condition_num 5 --optimal --surface --contour
