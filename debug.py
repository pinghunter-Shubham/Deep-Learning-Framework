import traceback
from tests.test_gradcheck import *

tests_to_run = [
    test_grad_check_add, 
    test_grad_check_broadcast_add, 
    test_grad_check_mul, 
    test_grad_check_sub, 
    test_grad_check_matmul, 
    test_grad_check_reshape, 
    test_grad_check_transpose, 
    test_grad_check_complex_graph
]

for test in tests_to_run:
    try:
        print(f"---- Running {test.__name__} ----")
        test()
        print("Success")
    except Exception as e:
        traceback.print_exc()
