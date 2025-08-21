# Notes
## Test
```
>> pip install pytest
```

```
>> export PYTHONPATH="src"

>> pytest
>> pytest tests/function_test.py
>> pytest tests/function_test.py::test_add
>> pytest tests/function_test.py::test_add --trace
```

Run tests with output
```
>> pytest -s
```

Single step debugging
```
>> pytest --trace
```

## Miscellaneous Notes
### Numpy
#### Zero-Dimensional Arrays
```python
x = np.array([3.14]) # One-dimensional array
y = x ** 2
print(type(x), x.ndim)
print(type(y)) # 'numpy.ndarray'
```

```python
x = np.array(3.14)   # Zero-dimensional array
y = x ** 2
print(type(x), x.ndim)
print(type(y)) # 'numpy.float64'
```

### Python Debugging
```python
import pdb; pdb.set_trace()
```

### Windows PowerShell
#### Setting Up Environment Variables
```powershell
$env:PYTHONPATH="src"
```

Above is equivalen to below in Linux / macOS
```shell
export PYTHONPATH="src"
```

### Python Environment-Aware Path Setup
```python
if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
```

With this setup, both of the following commands will work:
```
>> python tests/simple_test.py
```

```
>> cd tests
>> python simple_test.py
```
