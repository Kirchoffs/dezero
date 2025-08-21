import os
import subprocess
import platform


def _dot_var(v, verbose = False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'

    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))

    return txt


def get_dot_graph(output, verbose = False):
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            
            if x.creator is not None:
                add_func(x.creator)

    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output, verbose = True, to_file = "graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")
    os.makedirs(tmp_dir, exist_ok = True)
    dot_graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(dot_graph_path, "w") as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]

    if platform.system() == "Windows":
        dot_graph_path = f'"{dot_graph_path}"'.replace('\\', '/')
        to_file = f'"{to_file}"'.replace('\\', '/')

    cmd = f"dot {dot_graph_path} -T {extension} -o {to_file}"
    subprocess.run(cmd, shell = True)


def reshape_sum_backward_for_broadcast(gy, x_shape, axis, keepdims):
    ndim = len(x_shape)
    
    tupled_axis = axis
    if tupled_axis is not None and not isinstance(tupled_axis, tuple):
        tupled_axis = (tupled_axis,)
    
    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [axis_index if axis_index >= 0 else axis_index + ndim for axis_index in tupled_axis]
        
        shape = list(gy.shape)
        for axis_index in sorted(actual_axis):
            shape.insert(axis_index, 1)
    else:
        shape = gy.shape

    return gy.reshape(shape)


def sum_to_shape(x, shape):
    ndim = len(shape)
    
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    
    y = x.sum(axis = lead_axis + axis, keepdims = True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    
    return y
