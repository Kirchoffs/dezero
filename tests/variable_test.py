from dezero import Variable


def test_variable_initialization_non_ndarray():
    try:
        Variable(2.718)
    except TypeError as e:
        print(e)
