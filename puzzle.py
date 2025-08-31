import streamlit as st
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# ============================================================
# Machinarium three-ring puzzle — faithful to the paper layout
#
# Geometry (from the TikZ figure in the paper):
#   R = 2.15
#   A = (-1.6, 0)
#   B = ( 1.6, 0)
#   C = ( 0.0, 2.4)
#
# Outer positions (six dots on the outside), by angle:
#   a1: 180°, a2: 240°
#   b1:   0°, b2: 300°
#   c1:  90°, c2:  30°
#
# Generators exactly as defined in the paper (cyclic order):
#   g_A = (a1 ca1 ab1 a2 ab2 ca2)
#   g_B = (b1 ab1 bc1 b2 bc2 ab2)
#   g_C = (c1 bc1 ca1 c2 ca2 bc2)
#
# UI: rotate A/B/C by one step (↻ = generator; ↺ = inverse),
#     undo, reset, shuffle (seedable), custom setup, auto-solve (BFS).
# ============================================================

# -----------------------------
# Geometry helpers
# -----------------------------

def circle_intersections(c0, r0, c1, r1):
    """Return (upper_point, lower_point) where 'upper' has larger y."""
    (x0, y0), (x1, y1) = c0, c1
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d == 0 or d > r0 + r1 or d < abs(r0 - r1):
        return None, None
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h_sq = max(0.0, r0**2 - a**2)
    h = math.sqrt(h_sq)
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    rx = -dy * (h / d)
    ry =  dx * (h / d)
    p_up = (xm + rx, ym + ry)
    p_dn = (xm - rx, ym - ry)
    if p_up[1] < p_dn[1]:
        p_up, p_dn = p_dn, p_up
    return p_up, p_dn

def pt_on_circle(center, R, deg):
    rad = math.radians(deg)
    return (center[0] + R * math.cos(rad), center[1] + R * math.sin(rad))

# -----------------------------
# Paper layout
# -----------------------------

def compute_layout():
    R = 2.15
    A = (-1.6, 0.0)
    B = ( 1.6, 0.0)
    C = ( 0.0, 2.4)

    # Intersections: label '1' = upper, '2' = lower (by y)
    ab1, ab2 = circle_intersections(A, R, B, R)   # A ∩ B
    ca1, ca2 = circle_intersections(C, R, A, R)   # C ∩ A
    bc1, bc2 = circle_intersections(B, R, C, R)   # B ∩ C

    coords = {
        # Intersections
        "ab1": ab1, "ab2": ab2,
        "bc1": bc1, "bc2": bc2,
        "ca1": ca1, "ca2": ca2,

        # Outer six at the paper's angles
        "a1": pt_on_circle(A, R, 180),
        "a2": pt_on_circle(A, R, 240),
        "b1": pt_on_circle(B, R,   0),
        "b2": pt_on_circle(B, R, 300),
        "c1": pt_on_circle(C, R,  120),
        "c2": pt_on_circle(C, R,  60),
    }

    return dict(A=A, B=B, C=C, R=R, coords=coords)

LAYOUT = compute_layout()

# -----------------------------
# Labels and goal
# -----------------------------

LABELS: Tuple[str, ...] = (
    "a1","a2","b1","b2","c1","c2","ab1","ab2","bc1","bc2","ca1","ca2"
)
OUTER: Tuple[str, ...] = ("a1","a2","b1","b2","c1","c2")
INDEX: Dict[str, int] = {lab: i for i, lab in enumerate(LABELS)}

# -----------------------------
# Generator cycles (from the paper)
# -----------------------------

CYCLES: Dict[str, Tuple[str, ...]] = {
    "A": ("a1","ca1","ab1","a2","ab2","ca2"),
    "B": ("b1","ab1","bc1","b2","bc2","ab2"),
    "C": ("c1","bc1","ca1","c2","ca2","bc2"),
}

def cycle_to_perm(cycle: Tuple[str, ...]) -> List[int]:
    """Permutation p where p[i] = destination index for bead currently at i."""
    p = list(range(len(LABELS)))
    for k in range(len(cycle)):
        src = INDEX[cycle[k]]
        dst = INDEX[cycle[(k + 1) % len(cycle)]]
        p[src] = dst
    return p

def invert_perm(p: List[int]) -> List[int]:
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return inv

PERM = {k: cycle_to_perm(v) for k, v in CYCLES.items()}
PERM_INV = {k + "'": invert_perm(PERM[k]) for k in CYCLES.keys()}
ALL_MOVES = list(PERM.keys()) + list(PERM_INV.keys())

# -----------------------------
# State & solver
# -----------------------------

def apply_perm(state: Tuple[int, ...], perm: List[int]) -> Tuple[int, ...]:
    new_state = [0] * len(state)
    for i, val in enumerate(state):
        new_state[perm[i]] = val
    return tuple(new_state)

def do_move(state: Tuple[int, ...], move: str) -> Tuple[int, ...]:
    if move in PERM:
        return apply_perm(state, PERM[move])
    elif move in PERM_INV:
        return apply_perm(state, PERM_INV[move])
    else:
        raise ValueError(f"Unknown move: {move}")

def is_goal(state: Tuple[int, ...]) -> bool:
    return all(state[INDEX[lbl]] == 1 for lbl in OUTER)

def scramble_from_solved(steps: int = 18, seed: int | None = None) -> Tuple[int, ...]:
    rng = random.Random(seed)
    state = tuple(1 if lab in OUTER else 0 for lab in LABELS)
    for _ in range(steps):
        state = do_move(state, rng.choice(ALL_MOVES))
    return state

@dataclass
class SearchResult:
    found: bool
    moves: List[str]

def bfs_min_solution(start: Tuple[int, ...]) -> SearchResult:
    if is_goal(start):
        return SearchResult(True, [])
    if sum(start) != 6:
        return SearchResult(False, [])

    q = deque([start])
    parent: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], str]] = {}
    seen = {start}

    while q:
        s = q.popleft()
        for mv in ALL_MOVES:
            ns = do_move(s, mv)
            if ns in seen:
                continue
            parent[ns] = (s, mv)
            if is_goal(ns):
                path = [mv]
                cur = s
                while cur != start:
                    prev, pmv = parent[cur]
                    path.append(pmv)
                    cur = prev
                path.reverse()
                return SearchResult(True, path)
            seen.add(ns)
            q.append(ns)
    return SearchResult(False, [])

# -----------------------------
# Drawing
# -----------------------------

def draw_state(state: Tuple[int, ...]):
    A, B, C, R = LAYOUT["A"], LAYOUT["B"], LAYOUT["C"], LAYOUT["R"]
    coords = LAYOUT["coords"]

    # bounds
    xs = [A[0], B[0], C[0]]
    ys = [A[1], B[1], C[1]]
    xmin, xmax = min(xs) - R - 0.4, max(xs) + R + 0.4
    ymin, ymax = min(ys) - R - 0.4, max(ys) + R + 0.4

    fig, ax = plt.subplots(figsize=(6.6, 6.8))

    # Circles
    ax.add_artist(plt.Circle(A, R, fill=False, linewidth=2))
    ax.add_artist(plt.Circle(B, R, fill=False, linewidth=2))
    ax.add_artist(plt.Circle(C, R, fill=False, linewidth=2))

    # Target halos on OUTER slots
    for lab in OUTER:
        x, y = coords[lab]
        ax.add_artist(plt.Circle((x, y), 0.135, fill=False, linewidth=1.2, linestyle='--', alpha=0.6))

    # Beads + labels
    for lab in LABELS:
        x, y = coords[lab]
        green = bool(state[INDEX[lab]])
        ax.add_artist(plt.Circle((x, y), 0.10, color=('#2ecc71' if green else '#bbbbbb'), ec='black', lw=0.6))
        ax.text(x + 0.06, y + 0.06, lab, fontsize=10, ha='left', va='bottom')

    # Circle labels near centers
    ax.text(A[0]-0.22, A[1]-0.25, 'A', fontsize=12)
    ax.text(B[0]+0.10, B[1]-0.25, 'B', fontsize=12)
    ax.text(C[0]+0.02, C[1]+0.28, 'C', fontsize=12)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis('off')
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Machinarium Three‑Ring Puzzle (paper layout)", layout="centered")

st.title("🔄 Machinarium Three‑Ring Puzzle — Paper Layout")
st.caption(
    "Rotate circles A, B, C. Goal: place the six green beads on the six dashed OUTER slots "
    "(a1, a2, b1, b2, c1, c2). Generators follow the paper: "
    "A=(a1 ca1 ab1 a2 ab2 ca2), B=(b1 ab1 bc1 b2 bc2 ab2), C=(c1 bc1 ca1 c2 ca2 bc2)."
)

# Session state
if 'state' not in st.session_state:
    st.session_state.state = scramble_from_solved(steps=18)
if 'history' not in st.session_state:
    st.session_state.history = []
if 'auto_solution' not in st.session_state:
    st.session_state.auto_solution = None

def push_history(move, prev_state):
    st.session_state.history.append((move, prev_state))

def do_and_record(move: str):
    prev = st.session_state.state
    st.session_state.state = do_move(prev, move)
    push_history(move, prev)

def reset_to_solved():
    st.session_state.state = tuple(1 if lab in OUTER else 0 for lab in LABELS)
    st.session_state.history = []
    st.session_state.auto_solution = None

def randomize(seed=None):
    st.session_state.state = scramble_from_solved(steps=25, seed=seed)
    st.session_state.history = []
    st.session_state.auto_solution = None

def undo():
    if st.session_state.history:
        _move, prev = st.session_state.history.pop()
        st.session_state.state = prev

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Rotate A")
    c1 = st.columns(2)
    if c1[0].button("A ↻", help="Apply g_A"):
        do_and_record('A')
    if c1[1].button("A ↺", help="Apply g_A⁻¹"):
        do_and_record("A'")
with col2:
    st.subheader("Rotate B")
    c2 = st.columns(2)
    if c2[0].button("B ↻", help="Apply g_B"):
        do_and_record('B')
    if c2[1].button("B ↺", help="Apply g_B⁻¹"):
        do_and_record("B'")
with col3:
    st.subheader("Rotate C")
    c3 = st.columns(2)
    if c3[0].button("C ↻", help="Apply g_C"):
        do_and_record('C')
    if c3[1].button("C ↺", help="Apply g_C⁻¹"):
        do_and_record("C'")

# Secondary controls
scol = st.columns(4)
if scol[0].button("Undo", help="Undo last move"):
    undo()
if scol[1].button("Reset to solved"):
    reset_to_solved()
seed_val = scol[3].number_input(
    "Seed", min_value=0, value=0, step=1,
    help="Optional: set a seed then click Shuffle for reproducibility."
)
if scol[2].button("Shuffle"):
    randomize(seed=int(seed_val))

# Custom setup
with st.expander("Custom setup (choose which 6 labels are green)", expanded=False):
    chosen = st.multiselect(
        "Pick exactly 6 labels to be green.",
        options=list(LABELS),
        default=[lab for lab, bit in zip(LABELS, st.session_state.state) if bit == 1]
    )
    cset = st.columns(2)
    if cset[0].button("Apply custom setup"):
        if len(chosen) != 6:
            st.warning("Please select exactly 6 labels.")
        else:
            st.session_state.state = tuple(1 if lab in set(chosen) else 0 for lab in LABELS)
            st.session_state.history = []
            st.session_state.auto_solution = None
    if cset[1].button("Apply solved (greens on OUTER)"):
        reset_to_solved()

# Draw
fig = draw_state(st.session_state.state)
st.pyplot(fig, use_container_width=True)

# Status and success banner
move_count = len(st.session_state.history)
st.info(f"Moves made: {move_count}")
if is_goal(st.session_state.state):
    st.success("Solved! All six green beads are on the OUTER slots.")

# -----------------------------
# Auto-solver
# -----------------------------

st.subheader("🧠 Auto‑solve (find a shortest solution)")
if sum(st.session_state.state) != 6:
    st.warning("Auto‑solver requires exactly 6 green beads in total. Use the custom setup to select exactly 6.")
else:
    if st.button("Find shortest solution from current position"):
        with st.spinner("Searching…"):
            res = bfs_min_solution(st.session_state.state)
        if not res.found:
            st.error("No solution found (this should not happen for valid setups with 6 greens).")
        else:
            st.session_state.auto_solution = res.moves
            st.success(
                f"Found solution in {len(res.moves)} moves: "
                f"{' '.join(res.moves) if res.moves else '(already solved)'}"
            )

    sol = st.session_state.get('auto_solution', None)
    if sol is not None:
        st.write("**Solution sequence**:", ' '.join(sol) if sol else '(already solved)')
        b1, b2 = st.columns(2)
        if b1.button("Apply entire solution"):
            s = st.session_state.state
            for mv in sol:
                push_history(mv, s)
                s = do_move(s, mv)
            st.session_state.state = s
        if b2.button("Step one move ▶"):
            if sol:
                mv = sol.pop(0)
                do_and_record(mv)
                st.session_state.auto_solution = sol
            else:
                st.info("No remaining steps; puzzle is solved.")

# Footer
st.caption(
    "This app uses the exact geometry and generator cycles from the paper: "
    "R=2.15; A=(-1.6,0); B=(1.6,0); C=(0,2.4); "
    "outer dots at a1=180°, a2=240°, b1=0°, b2=300°, c1=90°, c2=30°; "
    "g_A=(a1 ca1 ab1 a2 ab2 ca2), g_B=(b1 ab1 bc1 b2 bc2 ab2), g_C=(c1 bc1 ca1 c2 ca2 bc2)."
)
