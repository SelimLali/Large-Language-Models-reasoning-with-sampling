import re
from functools import lru_cache
from typing import List, Optional, Tuple

# -----------------------------
# 1) Robust boxed extraction
# -----------------------------
def _extract_braced(s: str, start_idx: int) -> Tuple[Optional[str], int]:
    """
    Extract {...} content with balanced braces.
    start_idx must point at '{'.
    Returns (content, end_idx_exclusive) where end_idx_exclusive is index after matching '}'.
    """
    if start_idx >= len(s) or s[start_idx] != "{":
        return None, start_idx

    depth = 0
    i = start_idx
    out = []
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out), i + 1
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None, start_idx


def extract_last_boxed(text: str) -> Optional[str]:
    """
    Return content of the last \\boxed{...} (balanced) in `text`.
    Also supports \\fbox{...}.
    """
    if not text:
        return None

    # find last occurrence among \boxed{ and \fbox{
    candidates = []
    for key in ("\\boxed{", "\\fbox{"):
        idx = text.rfind(key)
        if idx != -1:
            candidates.append((idx, key))
    if not candidates:
        return None

    idx, key = max(candidates, key=lambda x: x[0])
    brace_idx = idx + len(key) - 1  # points to '{'
    content, _ = _extract_braced(text, brace_idx)
    return content.strip() if content is not None else None


def extract_final_answer_from_completion(completion: str) -> str:
    """
    Best-effort extraction:
      1) last boxed content
      2) "Final answer: ..."
      3) "#### ..."
      4) last non-empty line
    """
    if not completion:
        return ""

    boxed = extract_last_boxed(completion)
    if boxed is not None and boxed.strip():
        return boxed.strip()

    m = re.search(r"(final answer|answer)\s*[:\-]\s*(.+)$", completion, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(2).strip()

    m2 = re.search(r"####\s*(.+)$", completion, flags=re.MULTILINE)
    if m2:
        return m2.group(1).strip()

    lines = [ln.strip() for ln in completion.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# -----------------------------
# 2) String cleanup helpers
# -----------------------------
_LATEX_WRAPPERS = [
    r"\left", r"\right",
]
_TEXT_CMD_RE = re.compile(r"\\text\{([^}]*)\}")
_RM_CMD_RE   = re.compile(r"\\mathrm\{([^}]*)\}")
_MB_CMD_RE   = re.compile(r"\\mathbf\{([^}]*)\}")

def strip_latex_wrappers(s: str) -> str:
    s = s.strip()
    s = s.replace("$", "")
    for w in _LATEX_WRAPPERS:
        s = s.replace(w, "")
    # unwrap common text-ish commands
    for _ in range(3):  # a few passes
        s = _TEXT_CMD_RE.sub(r"\1", s)
        s = _RM_CMD_RE.sub(r"\1", s)
        s = _MB_CMD_RE.sub(r"\1", s)
    # remove spacing commands
    s = s.replace("\\,", "").replace("\\ ", "")
    s = re.sub(r"\s+", "", s)
    return s

def normalize_text_answer(s: str) -> str:
    s = strip_latex_wrappers(s)
    # remove surrounding braces
    if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        s = s[1:-1]
    return s.lower()


# -----------------------------
# 3) Split at top-level commas (tuples/sets)
# -----------------------------
def split_top_level_commas(s: str) -> List[str]:
    parts = []
    cur = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


# -----------------------------
# 4) LaTeX -> sympy-ish conversion
# -----------------------------
def _replace_frac(s: str) -> str:
    # iterative replacement of \frac{a}{b} / \dfrac{a}{b}
    i = 0
    out = []
    while i < len(s):
        if s.startswith(r"\frac", i) or s.startswith(r"\dfrac", i):
            j = i + (5 if s.startswith(r"\frac", i) else 6)
            if j < len(s) and s[j] == "{":
                num, j2 = _extract_braced(s, j)
                if num is None:  # failed
                    out.append(s[i])
                    i += 1
                    continue
                if j2 < len(s) and s[j2] == "{":
                    den, j3 = _extract_braced(s, j2)
                    if den is None:
                        out.append(s[i])
                        i += 1
                        continue
                    out.append(f"(({num}))/(({den}))")
                    i = j3
                    continue
        out.append(s[i])
        i += 1
    return "".join(out)

def _replace_sqrt(s: str) -> str:
    # \sqrt{a} -> sqrt(a)
    while True:
        idx = s.find(r"\sqrt{")
        if idx == -1:
            break
        content, end = _extract_braced(s, idx + len(r"\sqrt"))
        if content is None:
            break
        # content already excludes outer braces
        s = s[:idx] + f"sqrt(({content}))" + s[end:]
    return s

def latex_to_sympy_str(s: str) -> str:
    s = strip_latex_wrappers(s)

    # degrees: 90^\circ -> 90*pi/180
    s = s.replace(r"^\circ", "*pi/180")

    # common symbols/operators
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = s.replace(r"\pi", "pi")
    s = s.replace("^", "**")

    # braces used for grouping -> parentheses
    s = s.replace("{", "(").replace("}", ")")

    s = _replace_frac(s)
    s = _replace_sqrt(s)

    # remove LaTeX spacing leftovers
    s = s.replace(r"\!", "")
    return s


# -----------------------------
# 5) Sympy parsing + equivalence
# -----------------------------
@lru_cache(maxsize=20000)
def _try_parse_sympy(expr: str):
    """
    Try parsing a math expression into sympy.
    Returns sympy expr or None.
    """
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application
        )
    except Exception:
        return None

    expr = expr.strip()
    if not expr:
        return None

    try:
        from sympy.parsing.latex import parse_latex 
        return parse_latex(expr)
    except Exception:
        pass

    # Second attempt: convert latex-ish to sympy string then parse_expr
    try:
        s2 = latex_to_sympy_str(expr)
        transformations = standard_transformations + (implicit_multiplication_application,)
        local_dict = {"pi": sp.pi, "sqrt": sp.sqrt}
        return parse_expr(s2, transformations=transformations, local_dict=local_dict, evaluate=True)
    except Exception:
        return None


def _is_number_like(s: str) -> bool:
    s = strip_latex_wrappers(s)
    return bool(re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", s))


def _float_value(s: str) -> Optional[float]:
    try:
        return float(strip_latex_wrappers(s))
    except Exception:
        return None


@lru_cache(maxsize=20000)
def is_equiv(pred: str, gold: str) -> bool:
    """
    Robust equivalence check for MATH answers.
    """
    if pred is None or gold is None:
        return False
    pred = pred.strip()
    gold = gold.strip()
    if not pred or not gold:
        return False

    # quick normalize
    p0 = strip_latex_wrappers(pred)
    g0 = strip_latex_wrappers(gold)

    # exact (after wrapper stripping)
    if p0 == g0:
        return True

    # text-like answers (names, words)
    # If gold contains letters and no digits/operators, compare as normalized text
    if re.search(r"[A-Za-z]", g0) and not re.search(r"[\d=+\-*/^]", g0):
        return normalize_text_answer(pred) == normalize_text_answer(gold)

    # tuples / comma-separated structures at top-level
    # e.g. (3, pi/2), {1,2,3}
    if (p0.startswith("(") and p0.endswith(")")) and (g0.startswith("(") and g0.endswith(")")):
        p_parts = split_top_level_commas(p0[1:-1])
        g_parts = split_top_level_commas(g0[1:-1])
        if len(p_parts) == len(g_parts) and len(p_parts) > 0:
            return all(is_equiv(pp, gg) for pp, gg in zip(p_parts, g_parts))

    if (p0.startswith("{") and p0.endswith("}")) and (g0.startswith("{") and g0.endswith("}")):
        # set comparison (order-insensitive)
        p_parts = split_top_level_commas(p0[1:-1])
        g_parts = split_top_level_commas(g0[1:-1])
        if len(p_parts) == len(g_parts) and len(p_parts) > 0:
            used = [False] * len(g_parts)
            for pp in p_parts:
                ok = False
                for j, gg in enumerate(g_parts):
                    if not used[j] and is_equiv(pp, gg):
                        used[j] = True
                        ok = True
                        break
                if not ok:
                    return False
            return True

    # numeric fast path
    if _is_number_like(pred) and _is_number_like(gold):
        pv = _float_value(pred)
        gv = _float_value(gold)
        if pv is not None and gv is not None:
            return abs(pv - gv) <= 1e-9

    # sympy equivalence
    p_expr = _try_parse_sympy(pred)
    g_expr = _try_parse_sympy(gold)
    if p_expr is not None and g_expr is not None:
        try:
            import sympy as sp
            diff = sp.simplify(p_expr - g_expr)
            if diff == 0:
                return True
        except Exception:
            pass

    # fallback: remove trivial parentheses then compare
    def strip_parens(x: str) -> str:
        x = strip_latex_wrappers(x)
        if len(x) >= 2 and x[0] == "(" and x[-1] == ")":
            return x[1:-1]
        return x

    return strip_parens(pred) == strip_parens(gold)
