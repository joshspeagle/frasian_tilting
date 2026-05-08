"""Per-method illustrative demo scripts.

Each registered method has an `<name>_demo.py` here that runs in
`--smoke` mode under CI to verify the method produces a figure
end-to-end. The CI workflow `method-completeness.yaml` exercises
each demo; `tools/check_method_completeness.py` enforces the
brief / property-test / illustration triple per registered method.
"""
