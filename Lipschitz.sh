#/usr/local/bin/python3
# Empirical Gradient
python3 main.py --function QUADRATIC  --function_param 2 --condition_num 1 --optimal --surface --contour --step_function LIPSCHITZ
python3 main.py --function LOG --function_param 0.1 --condition_num 1 --optimal --surface --contour --step_function LIPSCHITZ

python3 main.py --function QUADRATIC  --function_param 0.2 --condition_num 5 --optimal --surface --contour --step_function LIPSCHITZ
python3 main.py --function LOG --function_param 0.0001 --condition_num 5 --optimal --surface --contour --step_function LIPSCHITZ

# True Gradient
python3 main.py --function QUADRATIC  --function_param 2 --condition_num 1 --optimal --surface --contour --true_gradient --step_function LIPSCHITZ
python3 main.py --function LOG --function_param 0.1 --condition_num 1 --optimal --surface --contour --true_gradient --step_function LIPSCHITZ

python3 main.py --function QUADRATIC  --function_param 0.2 --condition_num 5 --optimal --surface --contour --true_gradient --step_function LIPSCHITZ
python3 main.py --function LOG --function_param 0.0001 --condition_num 5 --optimal --surface --contour --true_gradient --step_function LIPSCHITZ