if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


from dezero import Variable


def test_variable_initialization_non_ndarray():
    try:
        Variable(2.718)
    except TypeError as e:
        print(e)
