"""
Deterministic trace instrumentation — Phase 2 support.

Prompting the Coder to add its own debug prints turned out to be
unreliable: a small local model juggling several instructions at once
often just drops it, leaving the Verifier's captured stdout empty and the
trace analyzer with nothing to work from. This module rewrites the
generated solution's AST instead, guaranteeing: after every loop-body
variable update, and before every return, the code prints the value —
regardless of whether the model remembered to.

Falls back to the original source untouched on any parse/transform
failure; tracing is a nice-to-have; it must never be able to break a
solution that would otherwise have been correct.
"""

from __future__ import annotations

import ast


def _print_stmt(label: str, value_expr: ast.expr) -> ast.Expr:
    fstring = ast.JoinedStr(
        values=[
            ast.Constant(value=f"{label}="),
            ast.FormattedValue(value=value_expr, conversion=-1, format_spec=None),
        ]
    )
    call = ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[fstring], keywords=[])
    return ast.Expr(value=call)


def _print_const(text: str) -> ast.Expr:
    call = ast.Call(
        func=ast.Name(id="print", ctx=ast.Load()),
        args=[ast.Constant(value=text)],
        keywords=[],
    )
    return ast.Expr(value=call)


def _flatten_targets(target: ast.expr) -> list[tuple[ast.expr, str]]:
    """Yield (readable-name-node, label) pairs for Name/Tuple/List/Starred
    targets. Attribute/Subscript targets are skipped — re-evaluating
    `obj[i]` or `obj.attr` just to print it risks side effects, so only
    plain variables are traced."""
    if isinstance(target, ast.Name):
        return [(ast.Name(id=target.id, ctx=ast.Load()), target.id)]
    if isinstance(target, (ast.Tuple, ast.List)):
        pairs: list[tuple[ast.expr, str]] = []
        for elt in target.elts:
            pairs.extend(_flatten_targets(elt))
        return pairs
    if isinstance(target, ast.Starred):
        return _flatten_targets(target.value)
    return []


def _prints_for_assign(stmt: ast.stmt) -> list[ast.stmt]:
    targets: list[ast.expr] = []
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, ast.AugAssign):
        targets = [stmt.target]
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        targets = [stmt.target]
    else:
        return []
    prints: list[ast.stmt] = []
    for t in targets:
        for name_node, label in _flatten_targets(t):
            prints.append(_print_stmt(label, name_node))
    return prints


class _Instrumenter:
    def __init__(self) -> None:
        self._ret_counter = 0

    def _return_replacement(self, stmt: ast.Return) -> list[ast.stmt]:
        if stmt.value is None:
            return [_print_const("return=None"), stmt]
        tmp = f"__dbg_ret_{self._ret_counter}"
        self._ret_counter += 1
        assign = ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=stmt.value)
        print_stmt = _print_stmt("return", ast.Name(id=tmp, ctx=ast.Load()))
        new_return = ast.Return(value=ast.Name(id=tmp, ctx=ast.Load()))
        return [assign, print_stmt, new_return]

    def _instrument_stmt(self, stmt: ast.stmt, in_loop: bool) -> list[ast.stmt]:
        if isinstance(stmt, ast.Return):
            return self._return_replacement(stmt)

        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            stmt.body = self._instrument_block(stmt.body, True)
            if stmt.orelse:
                stmt.orelse = self._instrument_block(stmt.orelse, True)
            return [stmt]

        if isinstance(stmt, ast.If):
            stmt.body = self._instrument_block(stmt.body, in_loop)
            if stmt.orelse:
                stmt.orelse = self._instrument_block(stmt.orelse, in_loop)
            return [stmt]

        if isinstance(stmt, ast.Try):
            stmt.body = self._instrument_block(stmt.body, in_loop)
            for handler in stmt.handlers:
                handler.body = self._instrument_block(handler.body, in_loop)
            if stmt.orelse:
                stmt.orelse = self._instrument_block(stmt.orelse, in_loop)
            if stmt.finalbody:
                stmt.finalbody = self._instrument_block(stmt.finalbody, in_loop)
            return [stmt]

        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            stmt.body = self._instrument_block(stmt.body, in_loop)
            return [stmt]

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # New scope — instrument its own body independently, starting
            # fresh (not "in a loop" just because the enclosing code was).
            stmt.body = self._instrument_block(stmt.body, False)
            return [stmt]

        result: list[ast.stmt] = [stmt]
        if in_loop:
            result.extend(_prints_for_assign(stmt))
        return result

    def _instrument_block(self, stmts: list[ast.stmt], in_loop: bool) -> list[ast.stmt]:
        new_stmts: list[ast.stmt] = []
        for stmt in stmts:
            new_stmts.extend(self._instrument_stmt(stmt, in_loop))
        return new_stmts

    def instrument(self, tree: ast.Module) -> ast.Module:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.body = self._instrument_block(node.body, False)
        return tree


def instrument_source(source: str) -> str:
    """Return `source` with print() tracing inserted after every loop-body
    variable update and before every return. Returns `source` unchanged if
    it fails to parse, transform, or re-validate."""
    try:
        tree = ast.parse(source)
        tree = _Instrumenter().instrument(tree)
        ast.fix_missing_locations(tree)
        new_source = ast.unparse(tree)
        compile(new_source, "<instrumented>", "exec")
        return new_source
    except Exception:
        return source
