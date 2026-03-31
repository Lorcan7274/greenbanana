"""
KI1199 Molecular Docking GUI
A research-grade GUI for enzyme-ligand docking prediction.
Requires: ttkbootstrap, tkinter, matplotlib, numpy
Optional: rdkit, py3Dmol (for advanced features)

Install dependencies:
    pip install ttkbootstrap matplotlib numpy
    pip install rdkit-pypi  (optional, for molecule drawing)
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk as tk_ttk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
try:
    from ttkbootstrap.widgets.scrolled import ScrolledFrame
except ImportError:
    from ttkbootstrap.scrolled import ScrolledFrame
import threading
import time
import random
import math
import os
import json
from datetime import datetime

# ── Try optional heavy imports ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS & THEME
# ═══════════════════════════════════════════════════════════════════════════

APP_TITLE  = "KI1199 Docking Suite"
APP_THEME  = "darkly"          # ttkbootstrap dark theme
ACCENT     = "#00d4aa"         # teal accent
ACCENT2    = "#ff6b6b"         # coral for warnings
BG_CARD    = "#1e2a3a"         # card background
BG_DARK    = "#141b27"         # main background

PRESET_LIGANDS = {
    "ATP (Adenosine Triphosphate)": {"smiles": "C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N", "mw": 507.2},
    "Ibuprofen":                    {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                                              "mw": 206.3},
    "Aspirin":                      {"smiles": "CC(=O)Oc1ccccc1C(=O)O",                                                   "mw": 180.2},
    "Caffeine":                     {"smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",                                              "mw": 194.2},
    "Penicillin G":                 {"smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)Cc3ccccc3)C(=O)O)C",                           "mw": 334.4},
    "Quercetin":                    {"smiles": "c1c(cc2c(c1O)C(=O)c(c(o2)O)c3ccc(c(c3)O)O)O",                            "mw": 302.2},
    "Resveratrol":                  {"smiles": "c1cc(ccc1/C=C/c2cc(cc(c2)O)O)O",                                          "mw": 228.2},
    "Curcumin":                     {"smiles": "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",                      "mw": 368.4},
}

RESIDUES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL"]

MUTATIONS = ["A45T", "G89D", "K120R", "L156V", "R203Q",
             "N78S", "E134K", "H201Y", "C55A", "F99L"]


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATED DOCKING ENGINE  (replace with AutoDock Vina subprocess calls)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_docking(ligand_name, params, mutation=None, progress_cb=None):
    """
    Simulate docking computation.
    Replace this function body with subprocess calls to AutoDock Vina:
        vina --receptor protein.pdbqt --ligand ligand.pdbqt
             --center_x X --center_y Y --center_z Z
             --size_x Sx --size_y Sy --size_z Sz
             --exhaustiveness E --num_modes N
             --out output.pdbqt --log vina.log
    """
    steps = 20
    for i in range(steps):
        time.sleep(0.12)
        if progress_cb:
            progress_cb(int((i + 1) / steps * 100))

    base = random.uniform(-10.5, -5.0)
    if mutation:
        delta = random.uniform(-1.5, 1.5)
    else:
        delta = 0.0

    poses = []
    for rank in range(1, params["num_poses"] + 1):
        score = base + (rank - 1) * random.uniform(0.4, 0.9) + delta
        poses.append({
            "rank":  rank,
            "score": round(score, 2),
            "rmsd_lb": round(random.uniform(0, 2.0), 2),
            "rmsd_ub": round(random.uniform(2.0, 5.0), 2),
        })

    # Interaction summary
    n_hbonds  = random.randint(2, 6)
    n_vdw     = random.randint(3, 10)
    n_hphob   = random.randint(1, 4)
    residues  = random.sample(RESIDUES, min(5, len(RESIDUES)))
    distances = [round(random.uniform(2.5, 4.5), 2) for _ in residues]

    interactions = {
        "hbonds":       n_hbonds,
        "vdw":          n_vdw,
        "hydrophobic":  n_hphob,
        "residues":     residues,
        "distances":    distances,
    }

    score_wt = base
    score_mt = base + delta if mutation else None

    return {
        "ligand":       ligand_name,
        "mutation":     mutation,
        "poses":        poses,
        "interactions": interactions,
        "score_wt":     round(score_wt, 2),
        "score_mt":     round(score_mt, 2) if score_mt is not None else None,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def interpret_score(score):
    if score <= -9.0:
        return "🟢 Very strong binding — likely active site engagement with multiple key residues.", SUCCESS
    elif score <= -7.0:
        return "🟡 Moderate–strong binding — promising candidate for further optimization.", WARNING
    elif score <= -5.0:
        return "🟠 Weak–moderate binding — limited interactions, poor shape complementarity.", WARNING
    else:
        return "🔴 Poor binding affinity — ligand unlikely to be a viable inhibitor.", DANGER


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class KI1199App(ttk.Window):
    def __init__(self):
        super().__init__(themename=APP_THEME)
        self.title(APP_TITLE)
        self.geometry("1280x820")
        self.minsize(1100, 720)
        self._configure_styles()
        self._build_header()
        self._build_notebook()
        self._build_statusbar()
        self.results_history = []

    # ── Styles ──────────────────────────────────────────────────────────────
    def _configure_styles(self):
        s = ttk.Style()
        s.configure("Card.TFrame",        background=BG_CARD)
        s.configure("Title.TLabel",       font=("Courier New", 22, "bold"), foreground=ACCENT)
        s.configure("SectionHead.TLabel", font=("Courier New", 12, "bold"), foreground=ACCENT)
        s.configure("Mono.TLabel",        font=("Courier New", 10))
        s.configure("Score.TLabel",       font=("Courier New", 18, "bold"), foreground=ACCENT)
        s.configure("Accent.TButton",     font=("Courier New", 10, "bold"))

    # ── Header ───────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = ttk.Frame(self, padding=(20, 12))
        hdr.pack(fill=X)

        ttk.Label(hdr, text="⚗  KI1199 DOCKING SUITE",
                  style="Title.TLabel").pack(side=LEFT)

        ttk.Label(hdr,
                  text="Molecular docking & binding affinity prediction  |  AutoDock Vina pipeline",
                  font=("Courier New", 9), foreground="#7a8fa6").pack(side=LEFT, padx=20)

        self.clock_var = tk.StringVar()
        ttk.Label(hdr, textvariable=self.clock_var,
                  font=("Courier New", 9), foreground="#7a8fa6").pack(side=RIGHT)
        self._tick_clock()

    def _tick_clock(self):
        self.clock_var.set(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ── Notebook (tabs) ──────────────────────────────────────────────────────
    def _build_notebook(self):
        self.nb = ttk.Notebook(self, bootstyle=PRIMARY)
        self.nb.pack(fill=BOTH, expand=True, padx=10, pady=(0, 6))

        self.tab_dock     = self._make_docking_tab()
        self.tab_screen   = self._make_screening_tab()
        self.tab_mutation = self._make_mutation_tab()
        self.tab_heatmap  = self._make_heatmap_tab()
        self.tab_history  = self._make_history_tab()

        self.nb.add(self.tab_dock,     text="  ⚛  Docking  ")
        self.nb.add(self.tab_screen,   text="  🔬 Screening  ")
        self.nb.add(self.tab_mutation, text="  🧬 Mutation Combo  ")
        self.nb.add(self.tab_heatmap,  text="  📊 Heatmap  ")
        self.nb.add(self.tab_history,  text="  📋 History  ")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1 — DOCKING
    # ══════════════════════════════════════════════════════════════════════
    def _make_docking_tab(self):
        root = ttk.Frame(self.nb)
        root.columnconfigure(0, weight=1, minsize=360)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)

        # ── Left panel: inputs ──────────────────────────────────────────
        left = ScrolledFrame(root, autohide=True)
        left.grid(row=0, column=0, sticky=NSEW, padx=(8, 4), pady=8)
        lf = left.container

        # Enzyme section
        self._section(lf, "① ENZYME STRUCTURE (KI1199)")
        self.pdb_path = tk.StringVar(value="No file selected")
        ttk.Entry(lf, textvariable=self.pdb_path, state="readonly",
                  font=("Courier New", 9)).pack(fill=X, pady=(0, 4))
        ttk.Button(lf, text="📂  Upload .pdb File", bootstyle=SECONDARY,
                   command=self._load_pdb).pack(fill=X)
        ttk.Separator(lf).pack(fill=X, pady=10)

        # Ligand section
        self._section(lf, "② LIGAND INPUT")
        self.lig_mode = tk.StringVar(value="preset")
        modes = [("Preset Library", "preset"), ("Upload .mol/.sdf", "upload"), ("SMILES String", "smiles")]
        for txt, val in modes:
            ttk.Radiobutton(lf, text=txt, value=val, variable=self.lig_mode,
                            command=self._toggle_lig_mode).pack(anchor=W)

        # Preset picker
        self.preset_frame = ttk.Frame(lf)
        self.preset_frame.pack(fill=X, pady=4)
        ttk.Label(self.preset_frame, text="Select ligand:", font=("Courier New", 9)).pack(anchor=W)
        self.preset_var = tk.StringVar(value=list(PRESET_LIGANDS.keys())[0])
        self.preset_cb  = ttk.Combobox(self.preset_frame, textvariable=self.preset_var,
                                        values=list(PRESET_LIGANDS.keys()),
                                        state="readonly", font=("Courier New", 9))
        self.preset_cb.pack(fill=X)

        # Upload frame (hidden initially)
        self.upload_frame = ttk.Frame(lf)
        self.lig_path = tk.StringVar(value="No file selected")
        ttk.Entry(self.upload_frame, textvariable=self.lig_path,
                  state="readonly", font=("Courier New", 9)).pack(fill=X, pady=(0, 4))
        ttk.Button(self.upload_frame, text="📂  Upload .mol / .sdf",
                   bootstyle=SECONDARY, command=self._load_ligand).pack(fill=X)

        # SMILES frame (hidden initially)
        self.smiles_frame = ttk.Frame(lf)
        ttk.Label(self.smiles_frame, text="Enter SMILES:", font=("Courier New", 9)).pack(anchor=W)
        self.smiles_var = tk.StringVar()
        ttk.Entry(self.smiles_frame, textvariable=self.smiles_var,
                  font=("Courier New", 9)).pack(fill=X)

        ttk.Separator(lf).pack(fill=X, pady=10)

        # Parameters
        self._section(lf, "③ DOCKING PARAMETERS")

        params_grid = ttk.Frame(lf)
        params_grid.pack(fill=X)
        params_grid.columnconfigure(1, weight=1)

        self.param_vars = {}
        param_defs = [
            ("Grid Box X (Å)",   "grid_x",         "20",  "Size of search space X"),
            ("Grid Box Y (Å)",   "grid_y",         "20",  "Size of search space Y"),
            ("Grid Box Z (Å)",   "grid_z",         "20",  "Size of search space Z"),
            ("Exhaustiveness",   "exhaustiveness", "8",   "Search depth (8=default, 32=thorough)"),
            ("Num Poses",        "num_poses",      "9",   "Number of binding poses to generate"),
            ("Energy Range",     "energy_range",   "3",   "Max energy diff from best (kcal/mol)"),
        ]
        for row_i, (label, key, default, tip) in enumerate(param_defs):
            ttk.Label(params_grid, text=label,
                      font=("Courier New", 9)).grid(row=row_i, column=0, sticky=W, pady=2)
            var = tk.StringVar(value=default)
            self.param_vars[key] = var
            ent = ttk.Entry(params_grid, textvariable=var, width=8,
                            font=("Courier New", 9))
            ent.grid(row=row_i, column=1, sticky=EW, padx=(8, 0), pady=2)
            ttk.Label(params_grid, text=tip,
                      font=("Courier New", 8), foreground="#7a8fa6").grid(
                          row=row_i, column=2, sticky=W, padx=(6, 0))

        ttk.Separator(lf).pack(fill=X, pady=10)

        # Run button + progress
        self.dock_btn = ttk.Button(lf, text="▶  RUN DOCKING",
                                   bootstyle=SUCCESS, command=self._run_docking)
        self.dock_btn.pack(fill=X, ipady=8)

        self.dock_progress = ttk.Progressbar(lf, bootstyle=SUCCESS+STRIPED,
                                             mode="determinate", maximum=100)
        self.dock_progress.pack(fill=X, pady=(6, 0))
        self.dock_status = tk.StringVar(value="Ready")
        ttk.Label(lf, textvariable=self.dock_status,
                  font=("Courier New", 9), foreground="#7a8fa6").pack(anchor=W)

        # ── Right panel: results ────────────────────────────────────────
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky=NSEW, padx=(4, 8), pady=8)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Score hero
        score_card = ttk.Frame(right)
        score_card.grid(row=0, column=0, sticky=EW, pady=(0, 6))
        score_card.columnconfigure(0, weight=1)
        ttk.Label(score_card, text=" 🧮 DOCKING SCORE ",
                  font=("Courier New", 9, "bold"), foreground=ACCENT).pack(anchor=W)
        ttk.Separator(score_card).pack(fill=X, pady=(2, 4))
        _sc_inner = ttk.Frame(score_card, padding=(14, 4))
        _sc_inner.pack(fill=X)
        _sc_inner.columnconfigure(1, weight=1)

        self.score_var = tk.StringVar(value="—")
        self.score_lbl = ttk.Label(_sc_inner, textvariable=self.score_var,
                                   style="Score.TLabel")
        self.score_lbl.grid(row=0, column=0, padx=(0, 20))

        self.interp_var = tk.StringVar(value="Run docking to see results.")
        ttk.Label(_sc_inner, textvariable=self.interp_var,
                  font=("Courier New", 10), wraplength=500,
                  justify=LEFT).grid(row=0, column=1, sticky=W)

        # Notebook inside results
        res_nb = ttk.Notebook(right, bootstyle=SECONDARY)
        res_nb.grid(row=1, column=0, sticky=NSEW)
        right.rowconfigure(1, weight=1)

        self.poses_tab    = self._make_poses_panel(res_nb)
        self.interact_tab = self._make_interact_panel(res_nb)
        self.chart_tab    = self._make_chart_panel(res_nb)

        res_nb.add(self.poses_tab,    text="  📍 Binding Poses  ")
        res_nb.add(self.interact_tab, text="  🔬 Interactions  ")
        res_nb.add(self.chart_tab,    text="  📈 Score Chart  ")

        return root

    def _make_poses_panel(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.poses_tree = ttk.Treeview(
            frame,
            columns=("rank", "score", "rmsd_lb", "rmsd_ub", "quality"),
            show="headings", bootstyle=INFO, height=10
        )
        for col, hdr, w in [
            ("rank",    "Rank",       60),
            ("score",   "Score (kcal/mol)", 160),
            ("rmsd_lb", "RMSD (lb)",  100),
            ("rmsd_ub", "RMSD (ub)",  100),
            ("quality", "Quality",    120),
        ]:
            self.poses_tree.heading(col, text=hdr)
            self.poses_tree.column(col, width=w, anchor=CENTER)

        self.poses_tree.grid(row=0, column=0, sticky=NSEW)
        sb = ttk.Scrollbar(frame, orient=VERTICAL, command=self.poses_tree.yview)
        sb.grid(row=0, column=1, sticky=NS)
        self.poses_tree.configure(yscrollcommand=sb.set)

        # Pose detail
        detail = ttk.Frame(frame)
        detail.grid(row=1, column=0, columnspan=2, sticky=EW, pady=(8, 0))
        ttk.Label(detail, text=" 🧠 INTERPRETATION ",
                  font=("Courier New", 9, "bold"), foreground=ACCENT).pack(anchor=W)
        ttk.Separator(detail).pack(fill=X, pady=(2, 4))
        _det_inner = ttk.Frame(detail, padding=(10, 4))
        _det_inner.pack(fill=X)
        self.pose_detail_var = tk.StringVar(value="Select a pose to view interpretation.")
        ttk.Label(_det_inner, textvariable=self.pose_detail_var,
                  font=("Courier New", 9), wraplength=620, justify=LEFT).pack(anchor=W)

        self.poses_tree.bind("<<TreeviewSelect>>", self._on_pose_select)
        return frame

    def _make_interact_panel(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # Summary cards row
        cards = ttk.Frame(frame)
        cards.grid(row=0, column=0, columnspan=2, sticky=EW, pady=(0, 10))
        for i in range(3):
            cards.columnconfigure(i, weight=1)

        self.hbond_var  = tk.StringVar(value="—")
        self.vdw_var    = tk.StringVar(value="—")
        self.hphob_var  = tk.StringVar(value="—")

        for col, label, var, style in [
            (0, "H-Bonds",       self.hbond_var, SUCCESS),
            (1, "Van der Waals", self.vdw_var,   INFO),
            (2, "Hydrophobic",   self.hphob_var, WARNING),
        ]:
            c = ttk.Frame(cards)
            c.grid(row=0, column=col, sticky=EW, padx=4)
            ttk.Label(c, text=f" {label} ",
                      font=("Courier New", 9, "bold"), foreground=ACCENT).pack(anchor=W)
            ttk.Separator(c).pack(fill=X, pady=(1, 3))
            _ci = ttk.Frame(c, padding=(10, 4))
            _ci.pack()
            ttk.Label(_ci, textvariable=var,
                      font=("Courier New", 22, "bold")).pack()

        # Residue table
        self.res_tree = ttk.Treeview(
            frame,
            columns=("residue", "distance", "type"),
            show="headings", bootstyle=SECONDARY, height=8
        )
        for col, hdr, w in [
            ("residue",  "Residue",         140),
            ("distance", "Distance (Å)",    140),
            ("type",     "Interaction",     200),
        ]:
            self.res_tree.heading(col, text=hdr)
            self.res_tree.column(col, width=w, anchor=CENTER)
        self.res_tree.grid(row=1, column=0, sticky=NSEW)

        sb2 = ttk.Scrollbar(frame, orient=VERTICAL, command=self.res_tree.yview)
        sb2.grid(row=1, column=1, sticky=NS)
        self.res_tree.configure(yscrollcommand=sb2.set)

        return frame

    def _make_chart_panel(self, parent):
        frame = ttk.Frame(parent)
        if HAS_MPL:
            self.fig = Figure(figsize=(6, 3.8), dpi=95, facecolor=BG_DARK)
            self.ax  = self.fig.add_subplot(111)
            self.ax.set_facecolor(BG_CARD)
            self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
            self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        else:
            ttk.Label(frame,
                      text="📦 matplotlib not installed.\nRun: pip install matplotlib numpy",
                      font=("Courier New", 11)).pack(expand=True)
        return frame

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2 — LIGAND SCREENING
    # ══════════════════════════════════════════════════════════════════════
    def _make_screening_tab(self):
        root = ttk.Frame(self.nb)
        root.columnconfigure(0, weight=1, minsize=300)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)

        # Left
        left = ScrolledFrame(root, autohide=True)
        left.grid(row=0, column=0, sticky=NSEW, padx=(8, 4), pady=8)
        lf = left.container

        self._section(lf, "SELECT LIGANDS TO SCREEN")
        ttk.Label(lf, text="Choose one or more ligands from the preset library:",
                  font=("Courier New", 9), wraplength=280).pack(anchor=W, pady=(0, 6))

        self.screen_vars = {}
        for lig in PRESET_LIGANDS:
            var = tk.BooleanVar(value=True)
            self.screen_vars[lig] = var
            ttk.Checkbutton(lf, text=lig, variable=var,
                            bootstyle=SUCCESS).pack(anchor=W, pady=1)

        ttk.Separator(lf).pack(fill=X, pady=10)
        self._section(lf, "SCREENING PARAMETERS")
        ttk.Label(lf, text="Exhaustiveness:", font=("Courier New", 9)).pack(anchor=W)
        self.screen_exhaust = tk.StringVar(value="8")
        ttk.Entry(lf, textvariable=self.screen_exhaust,
                  font=("Courier New", 9)).pack(fill=X, pady=(0, 6))

        ttk.Separator(lf).pack(fill=X, pady=10)
        self.screen_btn = ttk.Button(lf, text="▶  RUN SCREEN",
                                     bootstyle=SUCCESS, command=self._run_screening)
        self.screen_btn.pack(fill=X, ipady=8)
        self.screen_progress = ttk.Progressbar(lf, bootstyle=SUCCESS+STRIPED,
                                               mode="determinate", maximum=100)
        self.screen_progress.pack(fill=X, pady=(6, 0))
        self.screen_status = tk.StringVar(value="Ready")
        ttk.Label(lf, textvariable=self.screen_status,
                  font=("Courier New", 9), foreground="#7a8fa6").pack(anchor=W)

        # Right
        right = ttk.Frame(root, padding=10)
        right.grid(row=0, column=1, sticky=NSEW, padx=(4, 8), pady=8)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="🏆  RANKED LIGANDS",
                  style="SectionHead.TLabel").grid(row=0, column=0, sticky=W, pady=(0, 8))

        self.screen_tree = ttk.Treeview(
            right,
            columns=("rank", "ligand", "score", "quality", "mw"),
            show="headings", bootstyle=SUCCESS, height=15
        )
        for col, hdr, w in [
            ("rank",    "#",               50),
            ("ligand",  "Ligand",         250),
            ("score",   "Score (kcal/mol)", 160),
            ("quality", "Binding",         120),
            ("mw",      "MW (Da)",          90),
        ]:
            self.screen_tree.heading(col, text=hdr)
            self.screen_tree.column(col, width=w, anchor=CENTER)
        self.screen_tree.grid(row=1, column=0, sticky=NSEW)

        sb = ttk.Scrollbar(right, orient=VERTICAL, command=self.screen_tree.yview)
        sb.grid(row=1, column=1, sticky=NS)
        self.screen_tree.configure(yscrollcommand=sb.set)

        # Best candidate banner
        self.best_var = tk.StringVar(value="Run screening to find the best candidate.")
        best_card = ttk.Frame(right)
        best_card.grid(row=2, column=0, columnspan=2, sticky=EW, pady=(8, 0))
        ttk.Label(best_card, text=" 🥇 BEST CANDIDATE ",
                  font=("Courier New", 9, "bold"), foreground=ACCENT).pack(anchor=W)
        ttk.Separator(best_card).pack(fill=X, pady=(2, 4))
        _bc_inner = ttk.Frame(best_card, padding=(12, 4))
        _bc_inner.pack(fill=X)
        ttk.Label(_bc_inner, textvariable=self.best_var,
                  font=("Courier New", 11, "bold"), foreground=ACCENT,
                  wraplength=600).pack(anchor=W)

        return root

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 3 — MUTATION COMBO
    # ══════════════════════════════════════════════════════════════════════
    def _make_mutation_tab(self):
        root = ttk.Frame(self.nb)
        root.columnconfigure(0, weight=1, minsize=300)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)

        # Left
        left = ScrolledFrame(root, autohide=True)
        left.grid(row=0, column=0, sticky=NSEW, padx=(8, 4), pady=8)
        lf = left.container

        self._section(lf, "MUTATION ANALYSIS")
        ttk.Label(lf, text="Compare wild-type vs mutant KI1199 docking scores.",
                  font=("Courier New", 9), wraplength=280).pack(anchor=W, pady=(0, 8))

        ttk.Label(lf, text="Select Mutation:", font=("Courier New", 9)).pack(anchor=W)
        self.mut_var = tk.StringVar(value=MUTATIONS[0])
        ttk.Combobox(lf, textvariable=self.mut_var, values=MUTATIONS,
                     state="readonly", font=("Courier New", 9)).pack(fill=X, pady=(0, 8))

        ttk.Label(lf, text="Select Ligand:", font=("Courier New", 9)).pack(anchor=W)
        self.mut_lig_var = tk.StringVar(value=list(PRESET_LIGANDS.keys())[0])
        ttk.Combobox(lf, textvariable=self.mut_lig_var,
                     values=list(PRESET_LIGANDS.keys()),
                     state="readonly", font=("Courier New", 9)).pack(fill=X, pady=(0, 8))

        ttk.Separator(lf).pack(fill=X, pady=10)
        self.mut_btn = ttk.Button(lf, text="▶  COMPARE WT vs MUTANT",
                                  bootstyle=WARNING, command=self._run_mutation)
        self.mut_btn.pack(fill=X, ipady=8)
        self.mut_progress = ttk.Progressbar(lf, bootstyle=WARNING+STRIPED,
                                            mode="determinate", maximum=100)
        self.mut_progress.pack(fill=X, pady=(6, 0))
        self.mut_status = tk.StringVar(value="Ready")
        ttk.Label(lf, textvariable=self.mut_status,
                  font=("Courier New", 9), foreground="#7a8fa6").pack(anchor=W)

        # Right
        right = ttk.Frame(root, padding=10)
        right.grid(row=0, column=1, sticky=NSEW, padx=(4, 8), pady=8)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)

        ttk.Label(right, text="⚖  WT vs MUTANT COMPARISON",
                  style="SectionHead.TLabel").grid(row=0, column=0, columnspan=2,
                                                    sticky=W, pady=(0, 12))

        # WT card
        wt_card = ttk.Frame(right)
        wt_card.grid(row=1, column=0, sticky=EW, padx=(0, 6), pady=(0, 8))
        ttk.Label(wt_card, text=" WILD-TYPE KI1199 ",
                  font=("Courier New", 9, "bold"), foreground=ACCENT).pack(anchor=W)
        ttk.Separator(wt_card).pack(fill=X, pady=(2, 4))
        _wt_inner = ttk.Frame(wt_card, padding=(14, 4))
        _wt_inner.pack(fill=BOTH, expand=True)
        self.wt_score_var = tk.StringVar(value="—")
        ttk.Label(_wt_inner, textvariable=self.wt_score_var,
                  font=("Courier New", 20, "bold"), foreground=ACCENT).pack()
        self.wt_interp_var = tk.StringVar(value="")
        ttk.Label(_wt_inner, textvariable=self.wt_interp_var,
                  font=("Courier New", 9), wraplength=200).pack(pady=(4, 0))

        # Mutant card
        mt_card = ttk.Frame(right)
        mt_card.grid(row=1, column=1, sticky=EW, padx=(6, 0), pady=(0, 8))
        ttk.Label(mt_card, text=" MUTANT KI1199 ",
                  font=("Courier New", 9, "bold"), foreground="#ffca3a").pack(anchor=W)
        ttk.Separator(mt_card).pack(fill=X, pady=(2, 4))
        _mt_inner = ttk.Frame(mt_card, padding=(14, 4))
        _mt_inner.pack(fill=BOTH, expand=True)
        self.mt_score_var = tk.StringVar(value="—")
        ttk.Label(_mt_inner, textvariable=self.mt_score_var,
                  font=("Courier New", 20, "bold"), foreground="#ffca3a").pack()
        self.mt_interp_var = tk.StringVar(value="")
        ttk.Label(_mt_inner, textvariable=self.mt_interp_var,
                  font=("Courier New", 9), wraplength=200).pack(pady=(4, 0))

        # Delta card
        delta_card = ttk.Frame(right)
        delta_card.grid(row=2, column=0, columnspan=2, sticky=NSEW, pady=(0, 0))
        delta_card.columnconfigure(0, weight=1)
        ttk.Label(delta_card, text=" ΔΔBINDING ENERGY ",
                  font=("Courier New", 9, "bold"), foreground="#7a8fa6").pack(anchor=W)
        ttk.Separator(delta_card).pack(fill=X, pady=(2, 4))
        _delta_inner = ttk.Frame(delta_card, padding=(12, 4))
        _delta_inner.pack(fill=BOTH, expand=True)
        _delta_inner.columnconfigure(0, weight=1)
        self.delta_var   = tk.StringVar(value="")
        self.delta_interp = tk.StringVar(value="")
        ttk.Label(_delta_inner, textvariable=self.delta_var,
                  font=("Courier New", 16, "bold")).pack(pady=(4, 0))
        ttk.Label(_delta_inner, textvariable=self.delta_interp,
                  font=("Courier New", 10), wraplength=500).pack(pady=(8, 0))

        # Bar chart placeholder
        if HAS_MPL:
            self.mut_fig = Figure(figsize=(5, 2.5), dpi=90, facecolor=BG_DARK)
            self.mut_ax  = self.mut_fig.add_subplot(111)
            self.mut_ax.set_facecolor(BG_CARD)
            self.mut_canvas = FigureCanvasTkAgg(self.mut_fig, master=_delta_inner)
            self.mut_canvas.get_tk_widget().pack(fill=BOTH, expand=True, pady=(10, 0))

        return root

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 4 — HEATMAP
    # ══════════════════════════════════════════════════════════════════════
    def _make_heatmap_tab(self):
        root = ttk.Frame(self.nb, padding=10)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, sticky=EW, pady=(0, 10))

        ttk.Label(top, text="📊  BINDING AFFINITY HEATMAP",
                  style="SectionHead.TLabel").pack(side=LEFT)
        ttk.Button(top, text="🔄  Generate Heatmap", bootstyle=PRIMARY,
                   command=self._generate_heatmap).pack(side=RIGHT)

        ttk.Label(top,
                  text="X-axis: Ligands  |  Y-axis: Mutations  |  Values: Docking Score (kcal/mol)",
                  font=("Courier New", 9), foreground="#7a8fa6").pack(side=LEFT, padx=16)

        if HAS_MPL:
            self.heatmap_fig = Figure(figsize=(10, 5), dpi=90, facecolor=BG_DARK)
            self.heatmap_ax  = self.heatmap_fig.add_subplot(111)
            self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, master=root)
            self.heatmap_canvas.get_tk_widget().grid(row=1, column=0, sticky=NSEW)
            self._generate_heatmap()
        else:
            ttk.Label(root,
                      text="📦 matplotlib not installed.\nRun: pip install matplotlib numpy",
                      font=("Courier New", 12)).grid(row=1, column=0)

        return root

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 5 — HISTORY
    # ══════════════════════════════════════════════════════════════════════
    def _make_history_tab(self):
        root = ttk.Frame(self.nb, padding=10)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, sticky=EW, pady=(0, 10))
        ttk.Label(top, text="📋  EXPERIMENT HISTORY",
                  style="SectionHead.TLabel").pack(side=LEFT)
        ttk.Button(top, text="💾  Export JSON", bootstyle=SECONDARY,
                   command=self._export_json).pack(side=RIGHT, padx=4)
        ttk.Button(top, text="🗑  Clear", bootstyle=DANGER,
                   command=self._clear_history).pack(side=RIGHT)

        self.hist_tree = ttk.Treeview(
            root,
            columns=("time", "ligand", "mutation", "score", "poses"),
            show="headings", bootstyle=SECONDARY, height=20
        )
        for col, hdr, w in [
            ("time",     "Timestamp",       170),
            ("ligand",   "Ligand",          220),
            ("mutation", "Mutation",        120),
            ("score",    "Best Score",      140),
            ("poses",    "# Poses",          90),
        ]:
            self.hist_tree.heading(col, text=hdr)
            self.hist_tree.column(col, width=w, anchor=CENTER)
        self.hist_tree.grid(row=1, column=0, sticky=NSEW)

        sb = ttk.Scrollbar(root, orient=VERTICAL, command=self.hist_tree.yview)
        sb.grid(row=1, column=1, sticky=NS)
        self.hist_tree.configure(yscrollcommand=sb.set)

        return root

    # ── Status bar ──────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = ttk.Frame(self, padding=(10, 3))
        bar.pack(fill=X, side=BOTTOM)
        ttk.Separator(self).pack(fill=X, side=BOTTOM)
        self.status_var = tk.StringVar(value="✅  KI1199 Docking Suite ready.")
        ttk.Label(bar, textvariable=self.status_var,
                  font=("Courier New", 9), foreground="#7a8fa6").pack(side=LEFT)
        ttk.Label(bar, text="AutoDock Vina pipeline  |  KI1199 enzyme  |  © 2025",
                  font=("Courier New", 9), foreground="#7a8fa6").pack(side=RIGHT)

    # ══════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════
    def _section(self, parent, text):
        ttk.Label(parent, text=text,
                  style="SectionHead.TLabel").pack(anchor=W, pady=(10, 4))

    def _toggle_lig_mode(self):
        mode = self.lig_mode.get()
        self.preset_frame.pack_forget()
        self.upload_frame.pack_forget()
        self.smiles_frame.pack_forget()
        if mode == "preset":
            self.preset_frame.pack(fill=X, pady=4)
        elif mode == "upload":
            self.upload_frame.pack(fill=X, pady=4)
        else:
            self.smiles_frame.pack(fill=X, pady=4)

    def _load_pdb(self):
        path = filedialog.askopenfilename(
            title="Select PDB File",
            filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")]
        )
        if path:
            self.pdb_path.set(path)
            self.status_var.set(f"✅  Loaded enzyme: {os.path.basename(path)}")

    def _load_ligand(self):
        path = filedialog.askopenfilename(
            title="Select Ligand File",
            filetypes=[("MOL/SDF files", "*.mol *.sdf"), ("All files", "*.*")]
        )
        if path:
            self.lig_path.set(path)
            self.status_var.set(f"✅  Loaded ligand: {os.path.basename(path)}")

    def _get_params(self):
        try:
            return {
                "grid_x":         float(self.param_vars["grid_x"].get()),
                "grid_y":         float(self.param_vars["grid_y"].get()),
                "grid_z":         float(self.param_vars["grid_z"].get()),
                "exhaustiveness": int(self.param_vars["exhaustiveness"].get()),
                "num_poses":      int(self.param_vars["num_poses"].get()),
                "energy_range":   float(self.param_vars["energy_range"].get()),
            }
        except ValueError:
            messagebox.showerror("Invalid Parameters",
                                 "Please enter valid numeric values for all parameters.")
            return None

    def _get_ligand_name(self):
        mode = self.lig_mode.get()
        if mode == "preset":
            return self.preset_var.get()
        elif mode == "upload":
            p = self.lig_path.get()
            return os.path.basename(p) if p != "No file selected" else None
        else:
            s = self.smiles_var.get().strip()
            return f"Custom ({s[:20]}...)" if len(s) > 20 else f"Custom ({s})" if s else None

    # ══════════════════════════════════════════════════════════════════════
    #  DOCKING LOGIC
    # ══════════════════════════════════════════════════════════════════════
    def _run_docking(self):
        params = self._get_params()
        if not params:
            return
        lig = self._get_ligand_name()
        if not lig:
            messagebox.showwarning("No Ligand", "Please select or enter a ligand.")
            return

        self.dock_btn.configure(state=DISABLED)
        self.dock_progress["value"] = 0
        self.dock_status.set("Initializing docking engine…")
        self.status_var.set(f"⚙  Running docking for {lig}…")

        def _worker():
            def _prog(v):
                self.dock_progress["value"] = v
                self.dock_status.set(f"Computing… {v}%")
            result = simulate_docking(lig, params, progress_cb=_prog)
            self.after(0, lambda: self._display_docking(result))

        threading.Thread(target=_worker, daemon=True).start()

    def _display_docking(self, result):
        best = result["poses"][0]
        score = best["score"]
        text, style = interpret_score(score)

        self.score_var.set(f"{score}  kcal/mol")
        self.interp_var.set(text)

        # Poses table
        for row in self.poses_tree.get_children():
            self.poses_tree.delete(row)
        for p in result["poses"]:
            q = "Excellent" if p["score"] <= -9 else "Good" if p["score"] <= -7 else "Moderate" if p["score"] <= -5 else "Poor"
            self.poses_tree.insert("", END,
                values=(p["rank"], f"{p['score']} kcal/mol",
                        f"{p['rmsd_lb']} Å", f"{p['rmsd_ub']} Å", q))

        # Interactions
        ix = result["interactions"]
        self.hbond_var.set(str(ix["hbonds"]))
        self.vdw_var.set(str(ix["vdw"]))
        self.hphob_var.set(str(ix["hydrophobic"]))

        for row in self.res_tree.get_children():
            self.res_tree.delete(row)
        itypes = ["H-bond", "van der Waals", "Hydrophobic", "π-stacking", "Salt bridge"]
        for res, dist in zip(ix["residues"], ix["distances"]):
            itype = random.choice(itypes)
            self.res_tree.insert("", END, values=(res, f"{dist} Å", itype))

        # Chart
        if HAS_MPL:
            self._plot_poses(result["poses"])

        # History
        self._add_history(result)
        self.dock_btn.configure(state=NORMAL)
        self.dock_progress["value"] = 100
        self.dock_status.set("✅  Docking complete.")
        self.status_var.set(f"✅  Docking complete — Best score: {score} kcal/mol")

    def _plot_poses(self, poses):
        self.ax.clear()
        ranks  = [p["rank"]  for p in poses]
        scores = [p["score"] for p in poses]
        colors = [ACCENT if s <= -7 else "#ffca3a" if s <= -5 else ACCENT2 for s in scores]
        self.ax.barh(ranks, scores, color=colors, edgecolor="#ffffff22")
        self.ax.set_xlabel("Binding Affinity (kcal/mol)", color="#aab4c4",
                           fontfamily="monospace")
        self.ax.set_ylabel("Pose Rank", color="#aab4c4", fontfamily="monospace")
        self.ax.set_title("Docking Pose Scores", color=ACCENT,
                          fontfamily="monospace", fontsize=11)
        self.ax.tick_params(colors="#aab4c4")
        self.ax.invert_yaxis()
        self.ax.axvline(-7, color=ACCENT, linestyle="--", linewidth=0.8, alpha=0.6)
        self.ax.axvline(-9, color="#00ff88", linestyle="--", linewidth=0.8, alpha=0.6)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#2d3f54")
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_pose_select(self, event):
        sel = self.poses_tree.selection()
        if not sel:
            return
        vals = self.poses_tree.item(sel[0], "values")
        score_str = vals[1].split()[0]
        try:
            score = float(score_str)
        except ValueError:
            return
        text, _ = interpret_score(score)
        rank = vals[0]
        self.pose_detail_var.set(
            f"Pose #{rank}  —  {score} kcal/mol\n\n{text}\n\n"
            f"RMSD (lower bound): {vals[2]}   |   RMSD (upper bound): {vals[3]}\n\n"
            "Lower RMSD values indicate poses that are more distinct from each other. "
            "The top-ranked pose is the predicted binding conformation."
        )

    # ══════════════════════════════════════════════════════════════════════
    #  SCREENING LOGIC
    # ══════════════════════════════════════════════════════════════════════
    def _run_screening(self):
        selected = [k for k, v in self.screen_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("No Ligands", "Select at least one ligand.")
            return
        self.screen_btn.configure(state=DISABLED)
        self.screen_progress["value"] = 0
        self.screen_status.set("Screening…")

        def _worker():
            results = []
            params = {"num_poses": 3, "exhaustiveness": int(self.screen_exhaust.get())}
            for i, lig in enumerate(selected):
                def _prog(v, idx=i, total=len(selected)):
                    pct = int((idx / total * 100) + v / total)
                    self.screen_progress["value"] = pct
                    self.screen_status.set(f"Docking {lig[:30]}… ({idx+1}/{total})")
                r = simulate_docking(lig, params, progress_cb=_prog)
                results.append((lig, r["poses"][0]["score"]))
            results.sort(key=lambda x: x[1])
            self.after(0, lambda: self._display_screening(results))

        threading.Thread(target=_worker, daemon=True).start()

    def _display_screening(self, results):
        for row in self.screen_tree.get_children():
            self.screen_tree.delete(row)
        for rank, (lig, score) in enumerate(results, 1):
            q = "Excellent" if score <= -9 else "Good" if score <= -7 else "Moderate" if score <= -5 else "Poor"
            mw = PRESET_LIGANDS.get(lig, {}).get("mw", "—")
            tag = "best" if rank == 1 else ""
            self.screen_tree.insert("", END, values=(rank, lig, f"{score} kcal/mol", q, mw), tags=(tag,))

        self.screen_tree.tag_configure("best", foreground=ACCENT)
        best_lig, best_score = results[0]
        text, _ = interpret_score(best_score)
        self.best_var.set(f"{best_lig}  →  {best_score} kcal/mol\n{text}")

        self.screen_btn.configure(state=NORMAL)
        self.screen_progress["value"] = 100
        self.screen_status.set("✅  Screening complete.")
        self.status_var.set(f"✅  Screening done — Best: {best_lig} ({best_score} kcal/mol)")

    # ══════════════════════════════════════════════════════════════════════
    #  MUTATION LOGIC
    # ══════════════════════════════════════════════════════════════════════
    def _run_mutation(self):
        lig = self.mut_lig_var.get()
        mut = self.mut_var.get()
        self.mut_btn.configure(state=DISABLED)
        self.mut_progress["value"] = 0
        self.mut_status.set("Running WT docking…")

        def _worker():
            params = {"num_poses": 3, "exhaustiveness": 8}
            wt = simulate_docking(lig, params,
                                  progress_cb=lambda v: self.mut_progress.__setitem__("value", v // 2))
            self.mut_status.set(f"Running {mut} docking…")
            mt = simulate_docking(lig, params, mutation=mut,
                                  progress_cb=lambda v: self.mut_progress.__setitem__("value", 50 + v // 2))
            self.after(0, lambda: self._display_mutation(lig, mut, wt, mt))

        threading.Thread(target=_worker, daemon=True).start()

    def _display_mutation(self, lig, mut, wt, mt):
        ws = wt["poses"][0]["score"]
        ms = mt["poses"][0]["score"]
        delta = round(ms - ws, 2)

        self.wt_score_var.set(f"{ws} kcal/mol")
        self.mt_score_var.set(f"{ms} kcal/mol")
        wt_text, _ = interpret_score(ws)
        mt_text, _ = interpret_score(ms)
        self.wt_interp_var.set(wt_text)
        self.mt_interp_var.set(mt_text)

        sign = "+" if delta > 0 else ""
        self.delta_var.set(f"ΔΔG  =  {sign}{delta} kcal/mol")
        if delta > 0:
            interp = (f"Mutation {mut} WEAKENS binding by {abs(delta)} kcal/mol. "
                      "The substitution likely disrupts key contacts in the active site, "
                      "reducing inhibitor efficacy.")
        elif delta < 0:
            interp = (f"Mutation {mut} STRENGTHENS binding by {abs(delta)} kcal/mol. "
                      "The substitution may create additional favorable contacts, "
                      "potentially useful for drug-resistant variants.")
        else:
            interp = f"Mutation {mut} has negligible effect on binding affinity."
        self.delta_interp.set(interp)

        # Bar chart
        if HAS_MPL:
            self.mut_ax.clear()
            self.mut_ax.bar(["Wild-Type", f"Mutant ({mut})"], [ws, ms],
                            color=[ACCENT, "#ffca3a"], edgecolor="#ffffff22", width=0.4)
            self.mut_ax.set_ylabel("Binding Affinity (kcal/mol)", color="#aab4c4",
                                   fontfamily="monospace")
            self.mut_ax.set_title(f"{lig}  WT vs {mut}", color=ACCENT,
                                  fontfamily="monospace", fontsize=9)
            self.mut_ax.tick_params(colors="#aab4c4")
            for spine in self.mut_ax.spines.values():
                spine.set_edgecolor("#2d3f54")
            self.mut_fig.tight_layout()
            self.mut_canvas.draw()

        self.mut_btn.configure(state=NORMAL)
        self.mut_progress["value"] = 100
        self.mut_status.set("✅  Mutation comparison complete.")
        self.status_var.set(f"✅  {mut}: ΔΔG = {sign}{delta} kcal/mol")

    # ══════════════════════════════════════════════════════════════════════
    #  HEATMAP
    # ══════════════════════════════════════════════════════════════════════
    def _generate_heatmap(self):
        if not HAS_MPL:
            return
        ligands   = list(PRESET_LIGANDS.keys())
        mutations = MUTATIONS[:6]
        scores    = np.random.uniform(-10.5, -4.0, size=(len(mutations), len(ligands)))

        self.heatmap_ax.clear()
        im = self.heatmap_ax.imshow(scores, cmap="RdYlGn", aspect="auto",
                                    vmin=-11, vmax=-4)
        self.heatmap_fig.colorbar(im, ax=self.heatmap_ax, label="kcal/mol",
                                   fraction=0.03)
        self.heatmap_ax.set_xticks(range(len(ligands)))
        self.heatmap_ax.set_xticklabels(
            [l.split()[0] for l in ligands], rotation=30, ha="right",
            color="#aab4c4", fontfamily="monospace", fontsize=8
        )
        self.heatmap_ax.set_yticks(range(len(mutations)))
        self.heatmap_ax.set_yticklabels(mutations, color="#aab4c4",
                                         fontfamily="monospace", fontsize=9)
        self.heatmap_ax.set_title("Binding Affinity Heatmap — KI1199 Variants × Ligands",
                                   color=ACCENT, fontfamily="monospace", fontsize=11)

        for i in range(len(mutations)):
            for j in range(len(ligands)):
                self.heatmap_ax.text(j, i, f"{scores[i,j]:.1f}",
                                     ha="center", va="center",
                                     fontsize=7, color="white",
                                     fontfamily="monospace")

        self.heatmap_fig.tight_layout()
        self.heatmap_canvas.draw()

    # ══════════════════════════════════════════════════════════════════════
    #  HISTORY
    # ══════════════════════════════════════════════════════════════════════
    def _add_history(self, result):
        self.results_history.append(result)
        self.hist_tree.insert("", 0, values=(
            result["timestamp"],
            result["ligand"][:40],
            result["mutation"] or "Wild-type",
            f"{result['poses'][0]['score']} kcal/mol",
            len(result["poses"]),
        ))

    def _export_json(self):
        if not self.results_history:
            messagebox.showinfo("No Results", "Run some docking experiments first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"ki1199_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        if path:
            with open(path, "w") as f:
                json.dump(self.results_history, f, indent=2)
            self.status_var.set(f"✅  Exported {len(self.results_history)} results to {path}")

    def _clear_history(self):
        if messagebox.askyesno("Clear History", "Delete all experiment history?"):
            self.results_history.clear()
            for row in self.hist_tree.get_children():
                self.hist_tree.delete(row)
            self.status_var.set("🗑  History cleared.")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = KI1199App()
    app.mainloop()
