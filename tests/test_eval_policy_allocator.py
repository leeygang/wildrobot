import ast
from pathlib import Path


def test_standalone_eval_sets_platform_allocator_before_importing_jax():
    source = (
        Path(__file__).resolve().parents[1] / "training/eval/eval_policy.py"
    ).read_text()
    tree = ast.parse(source)
    allocator_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "setdefault"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "XLA_PYTHON_CLIENT_ALLOCATOR"
    )
    jax_import_line = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Import)
        and any(alias.name == "jax" for alias in node.names)
    )

    assert allocator_line < jax_import_line
