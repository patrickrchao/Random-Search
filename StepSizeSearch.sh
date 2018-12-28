#/usr/local/bin/python3
# Empirical Gradient
python3 main.py --function LOG --step_function INV_SQ_ROOT --search --function_param 0.1 --condition_num 1
python3 main.py --function LOG --step_function LOG --search --function_param 0.1 --condition_num 1
python3 main.py --function LOG --step_function GEOMETRIC --search --function_param 0.1 --condition_num 1
python3 main.py --function LOG --step_function CONSTANT --search --function_param 0.1 --condition_num 1

python3 main.py --function QUADRATIC --step_function INV_SQ_ROOT --search --function_param 2 --condition_num 1
python3 main.py --function QUADRATIC --step_function LOG --search --function_param 2 --condition_num 1
python3 main.py --function QUADRATIC --step_function GEOMETRIC --search --function_param 2 --condition_num 1
python3 main.py --function QUADRATIC --step_function CONSTANT --search --function_param 2 --condition_num 1


python3 main.py --function LOG --step_function INV_SQ_ROOT --search --function_param 0.0001 --condition_num 5
python3 main.py --function LOG --step_function LOG --search --function_param 0.0001 --condition_num 5
python3 main.py --function LOG --step_function GEOMETRIC --search --function_param 0.0001 --condition_num 5
python3 main.py --function LOG --step_function CONSTANT --search --function_param 0.0001 --condition_num 5

python3 main.py --function QUADRATIC --step_function INV_SQ_ROOT --search --function_param 0.2 --condition_num 5
python3 main.py --function QUADRATIC --step_function LOG --search --function_param 0.2 --condition_num 5
python3 main.py --function QUADRATIC --step_function GEOMETRIC --search --function_param 0.2 --condition_num 5
python3 main.py --function QUADRATIC --step_function CONSTANT --search --function_param 0.2 --condition_num 5


# True Gradient
python3 main.py --function LOG --step_function INV_SQ_ROOT --search --function_param 0.1 --condition_num 1 --true_gradient
python3 main.py --function LOG --step_function LOG --search --function_param 0.1 --condition_num 1 --true_gradient
python3 main.py --function LOG --step_function GEOMETRIC --search --function_param 0.1 --condition_num 1 --true_gradient
python3 main.py --function LOG --step_function CONSTANT --search --function_param 0.1 --condition_num 1 --true_gradient

python3 main.py --function QUADRATIC --step_function INV_SQ_ROOT --search --function_param 2 --condition_num 1 --true_gradient
python3 main.py --function QUADRATIC --step_function LOG --search --function_param 2 --condition_num 1 --true_gradient
python3 main.py --function QUADRATIC --step_function GEOMETRIC --search --function_param 2 --condition_num 1 --true_gradient
python3 main.py --function QUADRATIC --step_function CONSTANT --search --function_param 2 --condition_num 1 --true_gradient


python3 main.py --function LOG --step_function INV_SQ_ROOT --search --function_param 0.0001 --condition_num 5 --true_gradient
python3 main.py --function LOG --step_function LOG --search --function_param 0.0001 --condition_num 5 --true_gradient
python3 main.py --function LOG --step_function GEOMETRIC --search --function_param 0.0001 --condition_num 5 --true_gradient
python3 main.py --function LOG --step_function CONSTANT --search --function_param 0.0001 --condition_num 5 --true_gradient

python3 main.py --function QUADRATIC --step_function INV_SQ_ROOT --search --function_param 0.2 --condition_num 5 --true_gradient
python3 main.py --function QUADRATIC --step_function LOG --search --function_param 0.2 --condition_num 5 --true_gradient
python3 main.py --function QUADRATIC --step_function GEOMETRIC --search --function_param 0.2 --condition_num 5 --true_gradient
python3 main.py --function QUADRATIC --step_function CONSTANT --search --function_param 0.2 --condition_num 5 --true_gradient