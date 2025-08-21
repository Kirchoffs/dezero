# Notes
## Test
```
>> pip install pytest
```

```
>> pytest
>> pytest tests/function_test.py
>> pytest tests/function_test.py::test_add
```

Run tests with output
```
>> pytest -s
```

## Miscellaneous Notes
### Numpy
#### Zero-Dimensional Arrays
```python
x = np.array([3.14]) # One-dimensional array
y = x ** 2
print(type(x), x.ndim)
print(type(y)) # numpy.ndarray
```

```python
x = np.array(3.14)   # Zero-dimensional array
y = x ** 2
print(type(x), x.ndim)
print(type(y)) # numpy.float64
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
