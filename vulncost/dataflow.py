"""
Data-flow context for the target function, serialised as plain text.

Follows the "code with data flow" prompt of RevisitVD (Li et al., AsiaCCS '26,
Section 3.5), which in turn adapts GraphCodeBERT's data-flow graph: variable
occurrences are named by their order within the function ("the 2nd len"),
and each edge says which earlier occurrence a value comes from.

Simplification (state it in the Methods): edges follow source order, not
control flow. A use links to the most recent earlier definition of the same
name; branches are not merged. Only the target function is analysed, so this
adds no information from outside the function -- it restates what is already
in the code, in structured form.

    from vulncost.dataflow import data_flow_lines
    lines = data_flow_lines(func_source)        # list[str]
"""
from collections import defaultdict

import tree_sitter_c
from tree_sitter import Language, Parser

_PARSER = Parser(Language(tree_sitter_c.language()))
MAX_EDGES = 60   # cap so the context stays comparable in size to caller/callee


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _identifiers(node, out):
    """All identifier leaves under node, in source order. Skips called function
    names and struct field names, which are not variables."""
    if node.type == "identifier":
        out.append(node)
        return
    for i, child in enumerate(node.children):
        parent_field = node.field_name_for_child(i)
        if node.type == "call_expression" and parent_field == "function":
            continue
        if node.type == "field_expression" and parent_field == "field":
            continue
        _identifiers(child, out)


def _defined_name(node):
    """The identifier a declarator / lvalue ultimately names, or None."""
    while node is not None and node.type != "identifier":
        nxt = (node.child_by_field_name("declarator")
               or node.child_by_field_name("argument")
               or node.child_by_field_name("left"))
        if nxt is None:
            return None
        node = nxt
    return node


def data_flow_lines(func: str, max_edges: int = MAX_EDGES) -> list:
    src = func.encode("utf-8", "replace")
    tree = _PARSER.parse(src)
    text = lambda n: src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    # number every variable occurrence: (start_byte) -> "2nd len"
    occ = []
    _identifiers(tree.root_node, occ)
    seen = defaultdict(int)
    label = {}
    for n in occ:
        name = text(n)
        seen[name] += 1
        label[n.start_byte] = f"the {_ordinal(seen[name])} `{name}`"

    defs = set()      # start_byte of occurrences that are definitions
    computed = {}     # def start_byte -> [start_byte of source occurrences]

    def visit(node):
        t = node.type
        if t in ("init_declarator", "assignment_expression"):
            lhs = node.child_by_field_name("declarator") or node.child_by_field_name("left")
            rhs = node.child_by_field_name("value") or node.child_by_field_name("right")
            target = _defined_name(lhs)
            if target is not None and target.start_byte in label:
                defs.add(target.start_byte)
                srcs = []
                if rhs is not None:
                    _identifiers(rhs, srcs)
                # compound assignment (x += y) also reads x
                if t == "assignment_expression" and text(node.child_by_field_name("operator") or node) != "=":
                    srcs.insert(0, target)
                computed[target.start_byte] = [s.start_byte for s in srcs if s.start_byte in label]
        elif t == "update_expression":
            target = _defined_name(node.child_by_field_name("argument"))
            if target is not None and target.start_byte in label:
                defs.add(target.start_byte)
                computed[target.start_byte] = [target.start_byte]
        elif t == "parameter_declaration":
            target = _defined_name(node.child_by_field_name("declarator"))
            if target is not None and target.start_byte in label:
                defs.add(target.start_byte)
        for c in node.children:
            visit(c)

    visit(tree.root_node)

    edges = []
    last_def = {}     # name -> start_byte of most recent definition
    for n in occ:
        b, name = n.start_byte, text(n)
        if b in defs:
            for s in computed.get(b, []):
                if s != b:
                    edges.append(f"{label[b]} is computed from {label[s]}")
                elif name in last_def:
                    edges.append(f"{label[b]} is computed from {label[last_def[name]]}")
            last_def[name] = b
        elif name in last_def:
            edges.append(f"{label[b]} comes from {label[last_def[name]]}")

    # keep order, drop exact repeats, cap
    edges = list(dict.fromkeys(edges))
    if len(edges) > max_edges:
        edges = edges[:max_edges] + [f"... ({len(edges) - max_edges} more edges omitted)"]
    return edges


def render_data_flow(func: str) -> str:
    lines = data_flow_lines(func)
    parts = ["// Target function", func.rstrip(),
             "\n// ---- Context: data flow within the target function ----"]
    parts += [f"// {l}" for l in lines] if lines else ["// (no data flow found)"]
    return "\n".join(parts)