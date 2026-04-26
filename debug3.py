import traceback
from tests.test_gradcheck import test_grad_check_transpose

try:
    print("---- Running test_grad_check_transpose ----")
    test_grad_check_transpose()
    print("Success")
except Exception as e:
    traceback.print_exc()
