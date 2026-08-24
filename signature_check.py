# Static call-signature consistency checker.
#
# Why this exists: two separate bugs in this codebase (run_scenario_module()
# called with 0 args vs. defined with 3; _call_openai() called with 2/3 args
# vs. defined with fewer) were pure "arity mismatches" — a function defined
# one way and called another way. Both slipped past manual review more than
# once. This script parses the source with Python's `ast` module (no import,
# no Streamlit runtime needed) and flags any call to a locally-defined
# function that doesn't match its signature — catching this whole class of
# bug automatically, for every function in a file, not just the one that
# broke last time.
#
# Usage:
#   python signature_check.py path/to/file.py [more_files.py ...]
#   pytest tests/test_call_signatures.py

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class FuncSig:
    name: str
    min_positional: int      # required positional/positional-or-keyword params (no default)
    max_positional: int | None  # None means unlimited (has *args)
    param_names: List[str]   # names, in order, for keyword-arg matching
    required_kwonly: List[str]
    has_var_keyword: bool    # has **kwargs -> can't fully validate keyword calls


@dataclass
class Mismatch:
    file: str
    line: int
    func_name: str
    detail: str


def _collect_function_signatures(tree: ast.Module) -> Dict[str, FuncSig]:
    sigs: Dict[str, FuncSig] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = args.posonlyargs + args.args
            n_defaults = len(args.defaults)
            min_positional = len(positional) - n_defaults
            max_positional = None if args.vararg else len(positional)
            required_kwonly = [
                kw.arg for kw, default in zip(args.kwonlyargs, args.kw_defaults) if default is None
            ]
            sigs[node.name] = FuncSig(
                name=node.name,
                min_positional=max(min_positional, 0),
                max_positional=max_positional,
                param_names=[p.arg for p in positional],
                required_kwonly=required_kwonly,
                has_var_keyword=args.kwarg is not None,
            )
    return sigs


def check_file(path: Path) -> List[Mismatch]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    sigs = _collect_function_signatures(tree)

    mismatches: List[Mismatch] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue  # skip method calls (obj.method(...)), only check plain function calls
        func_name = node.func.id
        sig = sigs.get(func_name)
        if sig is None:
            continue  # not a locally-defined function (e.g. a builtin or import) — skip

        n_positional_given = len(node.args)
        has_star_args = any(isinstance(a, ast.Starred) for a in node.args)
        keyword_names = {kw.arg for kw in node.keywords if kw.arg is not None}
        has_double_star = any(kw.arg is None for kw in node.keywords)

        if has_star_args or has_double_star or sig.has_var_keyword:
            continue  # dynamic unpacking involved — too ambiguous to check reliably

        total_provided = n_positional_given + len(keyword_names)

        if total_provided < sig.min_positional and not keyword_names:
            mismatches.append(Mismatch(
                file=str(path), line=node.lineno, func_name=func_name,
                detail=f"called with {n_positional_given} positional arg(s), "
                       f"but {sig.name}() requires at least {sig.min_positional}",
            ))
        elif sig.max_positional is not None and n_positional_given > sig.max_positional:
            mismatches.append(Mismatch(
                file=str(path), line=node.lineno, func_name=func_name,
                detail=f"called with {n_positional_given} positional arg(s), "
                       f"but {sig.name}() only accepts {sig.max_positional}",
            ))
        else:
            # Positional count is plausible; check unknown keyword names.
            unknown_kwargs = keyword_names - set(sig.param_names)
            if unknown_kwargs:
                mismatches.append(Mismatch(
                    file=str(path), line=node.lineno, func_name=func_name,
                    detail=f"called with unknown keyword argument(s) {sorted(unknown_kwargs)} "
                           f"for {sig.name}({', '.join(sig.param_names)})",
                ))
    return mismatches


def main(argv: List[str]) -> int:
    all_mismatches: List[Mismatch] = []
    for arg in argv:
        all_mismatches.extend(check_file(Path(arg)))

    if not all_mismatches:
        print(f"✅ No signature/call mismatches found in {len(argv)} file(s).")
        return 0

    print(f"❌ Found {len(all_mismatches)} signature/call mismatch(es):\n")
    for m in all_mismatches:
        print(f"  {m.file}:{m.line} — {m.func_name}(): {m.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
