#/usr/local/bin/python3
python3 RandomSearchLipschitz.py --function_type 1 --step_function 1 --sigma_plots --surface_plots
python3 RandomSearchLipschitz.py --function_type 2 --step_function 1 --sigma_plots --surface_plots
python3 RandomSearchLipschitz.py --function_type 1 --step_function 2 --sigma_plots --surface_plots

python3 RandomSearchLipschitz.py --function_type 1 --step_function 1 --sigma_plots
python3 RandomSearchLipschitz.py --function_type 1 --step_function 2 --sigma_plots
python3 RandomSearchLipschitz.py --function_type 1 --step_function 3 --sigma_plots
python3 RandomSearchLipschitz.py --function_type 2 --step_function 1 --sigma_plots
python3 RandomSearchLipschitz.py --function_type 2 --step_function 2 --sigma_plots
python3 RandomSearchLipschitz.py --function_type 2 --step_function 3 --sigma_plots

python3 RandomSearchLipschitz.py --function_type 1 --step_function 1 --sigma_plots --param 2
python3 RandomSearchLipschitz.py --function_type 1 --step_function 2 --sigma_plots --param 2
python3 RandomSearchLipschitz.py --function_type 1 --step_function 3 --sigma_plots --param 2
python3 RandomSearchLipschitz.py --function_type 2 --step_function 1 --sigma_plots --param 0.01
python3 RandomSearchLipschitz.py --function_type 2 --step_function 2 --sigma_plots --param 0.01
python3 RandomSearchLipschitz.py --function_type 2 --step_function 3 --sigma_plots --param 0.01

python3 RandomSearchLipschitz.py --function_type 1 --step_function 1 --sigma_plots --param 2 --surface_plots
python3 RandomSearchLipschitz.py --function_type 2 --step_function 1 --sigma_plots --param 0.01 --surface_plots
