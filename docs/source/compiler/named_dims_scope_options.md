## Where should the named-dim annotations be scoped to?

### Option 1 — Per `torch.compile()` call (explicit re-annotation)

Annotations declared before a compile call are consumed by that compile and then discarded automatically. User must re-annotate before each compile call.

```python
declare_tensor_dim("B", B)
name_tensor_dims(x, ["B", "D"])
torch.compile(fn)(x)   # annotations used here, then gone

# Second compile — must re-annotate
declare_tensor_dim("B", B)
name_tensor_dims(x, ["B", "D"])
torch.compile(fn)(x)
```

### Option 2 — Per explicit user scope (context manager)

User wraps the annotate+compile block in a `with spyre_dims()` context manager. Annotations live for the duration of the `with` block and are cleared on exit. Supports multiple compiles in one scope.

```python
with spyre_dims():
    declare_tensor_dim("B", B)
    name_tensor_dims(x, ["B", "D"])
    torch.compile(fn)(x)   # annotations active
    torch.compile(fn2)(x)  # same annotations still active
# annotations cleared on __exit__
```

### Option 3 — Per `torch.compile()` call (inferred — no re-annotation needed)

Like option 1, but annotations are captured into the compiled function's closure at compile time. User only annotates once per model. Requires threading annotation state through Dynamo.

```python
declare_tensor_dim("B", B)
name_tensor_dims(x, ["B", "D"])
cfn = torch.compile(fn)   # annotations captured at compile time

cfn(x)   # works — annotations already captured
cfn(x)   # works again — no re-annotation needed
```

### Option 4b — Decorator with tensor mapping folded in

Both dim sizes and tensor-to-dim mapping declared on the function.

```python
@spyre_dims(B=64, D=128, tensors={"x": ["B", "D"]})
def fn(x):
    ...

cfn = torch.compile(fn)

cfn(x)   # all dim info comes from the decorator
cfn(x)   # same — no re-annotation needed
```

### Option 4c — Decorator with explicit `names` and `tensors` kwargs

Same as 4b but uses named kwargs `names=` and `tensors=` to separate the two concerns.

```python
@spyre_dims(names={"B": 64, "D": 128}, tensors={"x": ["B", "D"]})
def fn(x):
    ...

cfn = torch.compile(fn)

cfn(x)   # all dim info comes from the decorator
cfn(x)   # same — no re-annotation needed
```
