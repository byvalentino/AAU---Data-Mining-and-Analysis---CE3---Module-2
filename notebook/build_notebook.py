#!/usr/bin/env python3
"""Build Module 2's demonstration notebook, and execute it.

    python "Module 2/notebook/build_notebook.py"            build and run
    python "Module 2/notebook/build_notebook.py" --no-run   build only

Rewritten 28 September 2026, to the specification Module 4's notebook was rebuilt
to. The notebook follows the deck, `slides/Module2.pptx`, block by block, and
holds itself to three things:

  every image on a shown slide is drawn here by Python -- on the lab data where
      the slide shows data, from the same simulation or closed form where it
      shows one, and labelled "illustrative" where the slide's values were
      constructed and the lab data cannot stand in; photographs, logos,
      generated posters and copyrighted drawings are excluded with a reason
      (notebook/figure_map.json);
  every lab exercise is stated as its stub states it, and its solution runs with
      every function's code visible -- copied verbatim from exercises/solutions/,
      exercises/lab_support.py, exercises/_narrate.py and exercises/data/, and
      checked against those files by tools/check_notebook_sources.py;
  every number the slides print is recomputed, and agrees() stops the run if
      the deck and the code disagree. A number only the instructor's archive can
      produce is computed on the lab data and printed beside the recorded archive
      value (slides/measured.json, Module 1's measured.json), labelled "archive".

Data: only what the labs are given -- `exercises/data/bus_slice.csv.gz` (shuttle
VJRD1A10224000055, 22 and 23 January 2020, 48,290 readings, no personal data)
and the phone traces `exercises/data/prepare.py` generates from `make_phones.py`
and `calibration.json`. The archive files data/bus.csv and data/passengers.csv
are never opened. The notebook copies exercises/ to a temporary folder first and
works there, so running it never writes into the student's folder.

Executed from `Module 2/notebook`; it finds `../exercises` itself. Needs the lab
requirements plus notebook/requirements.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))

from notebook_kit import Notebook, execute  # noqa: E402

OUTPUT = HERE / "Module2_demonstration.ipynb"
EX = "Module 2/exercises"

nb = Notebook(2, HERE / "references.json")


# The shape of every explanation cell above a code cell.
def explain(goal: str, why: str, what: str, so_what: str, extra: str = "") -> None:
    text = (f"**Goal.** {goal}\n\n**Why.** {why}\n\n**What the code does.** {what}\n\n"
            f"**So what.** {so_what}")
    nb.md(text + (f"\n\n{extra}" if extra else ""))


# A copied file's path constants are computed from `__file__`; before each file is
# copied in, the notebook points `__file__` at it, in a cell of its own.
def repoint(relative: str) -> None:
    explain(
        f"Point `__file__` at `{relative}` in the working copy.",
        "The next cells are copied verbatim from that file, and its path constants are "
        "computed from `__file__`.",
        f"Sets `__file__` to the working copy's `{relative}`.",
        "The copied constants point where they point in the file itself.")
    nb.code(f"__file__ = str(Path.cwd() / {relative!r})")


# =============================================================================
# Front matter
# =============================================================================

def front_matter() -> None:
    nb.md("""
    # Module 2 — Cleaning data and building features

    **Data Mining and Analysis (course code CE3) · Aalborg University, Copenhagen**

    *How does raw data become correct model input — and what does each step cost?*

    The deck, `slides/Module2.pptx`, answers that question in four blocks, each
    followed by twenty-five minutes at the keyboard. This notebook is its companion:
    it follows the same four blocks in the same order, draws every figure the shown
    slides carry, states every laboratory exercise as the lab file states it, and
    runs the reference solution with all of its code on the page.

    | Block | The question | Laboratory |
    |---|---|---|
    | 1 | Know the distribution, then clean: what does a column's shape allow, and how do two clocks become one table? | Lab 1 — audit and align |
    | 2 | Missingness is a mechanism: why is a reading absent, and what does a fill invent? | Lab 2 — the mechanism |
    | 3 | Features, and the transform that is part of the model: which constants are learned, and which column already knows the answer? | Lab 3 — fit, and find the leak |
    | 4 | The series, and what it costs: windows, rhythms, a split that keeps time, and the price of a join | Lab 4 — windows and cost |

    **How to read it.** Every code cell has a note above it: the *goal*, *why* it is
    done, *what the code does*, and *so what* — what the result lets you say. Cells
    that begin `# Source: … verbatim` are copied from the laboratory files and checked
    against them, so the code you read is the code the labs run. Cells that draw a
    figure name the slide they reproduce, by its title. Formulas are written in Python
    with `sympy` and displayed as LaTeX; where it is cheap, a cell checks that the
    slide's formula and the lab's code are the same function.

    **Numbers.** `agrees()` prints a number a slide states beside the same number
    computed here, and stops the notebook if they differ. `beside()` prints two
    numbers that are expected to differ, with the reason — either because the slide
    quotes the instructor's archive, which this notebook does not open (those values
    are read from `slides/measured.json` and labelled *archive*), or because the deck
    has a slip, which is then said plainly.

    **Data.** Only what the labs receive: `exercises/data/bus_slice.csv.gz` — the
    real telemetry of shuttle VJRD1A10224000055 on 22 and 23 January 2020, 48,290
    readings, no personal data — and the phone traces that `exercises/data/prepare.py`
    generates from `make_phones.py`, whose parameters are the archive's measured
    magnitudes (`calibration.json`). The real phone file is the position record of
    sixteen identifiable volunteers; it is not in the repository and nothing here
    needs it.

    **How to run it.** From `Module 2/notebook` (or `Module 2/exercises`), in the lab
    environment (`bash setup.sh` in `exercises/`) plus
    `pip install -r ../notebook/requirements.txt`. The notebook copies `exercises/`
    to a temporary folder and works there. It runs in under a minute.

    **Beyond the deck.** The deck's one appendix, "The displacement budget, term by
    term", is worked in Block 2 beside the slide it belongs to.
    """)


# =============================================================================
# Set-up
# =============================================================================

def setup() -> None:
    nb.md("## Set-up")
    explain(
        "Load the libraries, and define the small tools every later cell uses.",
        "A notebook that claims to reproduce a deck needs a way to show a figure, a "
        "formula and a number side by side with what the slide printed.",
        "`show()` renders a plotly figure to a static PNG embedded in the notebook, so it "
        "displays anywhere, offline included. `formula()` displays sympy expressions as "
        "LaTeX. `agrees()` prints a number a slide states beside the same number computed "
        "here and stops the run when they differ; `beside()` prints two numbers that are "
        "expected to differ, with the reason. One warning is silenced: scikit-learn 1.5 "
        "passes an option newer SciPy no longer knows, and the warning names a local path.",
        "If the deck and the code ever disagree, this notebook fails to run rather than "
        "quietly showing a different number.")
    nb.code(r'''
    import json
    import math
    import os
    import shutil
    import sys
    import tempfile
    import types
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import sympy as sp
    from scipy import signal, stats
    from IPython.display import Image, Math, display

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Unknown solver options")
    warnings.filterwarnings("ignore", message="Could not infer format")
    pd.set_option("display.width", 120)

    BLUE, ORANGE, GREY, RED, GREEN, NAVY = ("#2A78D6", "#E07B39", "#52514E",
                                           "#C0392B", "#2E8B57", "#1F2A5A")


    def show(fig, name, width=1000, height=560):
        """Draw a plotly figure as a static image inside the notebook."""
        fig.update_layout(template="plotly_white", width=width, height=height)
        if fig.layout.margin.t is None:          # unless the figure asked for its own room
            fig.update_layout(margin=dict(l=70, r=30, t=70, b=60))
        display(Image(fig.to_image(format="png", width=width, height=height, scale=1)))


    def formula(*parts):
        """Display sympy expressions (or LaTeX strings) side by side."""
        pieces = [p if isinstance(p, str) else sp.latex(p, order="none") for p in parts]
        display(Math(r"\qquad ".join(pieces)))


    def agrees(what, computed, stated, places):
        """A number the deck states, recomputed here. Stops the run on a mismatch."""
        mine = round(float(computed), places)
        if abs(mine - float(stated)) > 0.5 * 10 ** -places + 1e-12:
            raise AssertionError(f"{what}: the deck says {stated}, the code gives {mine}")
        print(f"  {what:<66} deck {stated:<9} computed {int(mine) if places == 0 else mine}")


    DIFFERENCES = []   # every beside() call, gathered for the closing table


    def beside(what, computed, stated, why):
        """Two numbers that are expected to differ, printed with the reason."""
        DIFFERENCES.append((what, str(computed), str(stated), why))
        print(f"  {what}: computed here {computed}; stated {stated} — {why}")
    ''')

    nb.md("""
    ### A working copy of the exercises

    The labs write files: `data/*.parquet` and `data/MANIFEST.json` when the data are
    prepared, `out/handoff/` when Lab 4 hands its table on. A notebook that ran in the
    student's `exercises/` would leave its own copies there, and a student could not
    tell afterwards which files were theirs.
    """)
    explain(
        "Find `exercises/`, copy it to a temporary folder, and work inside the copy.",
        "The notebook must not modify the student's `exercises/`, and every relative path "
        "the labs use (`data/…`, `out/…`) must still resolve.",
        "Looks for the folder that holds `lab_support.py`, copies it without anything a "
        "previous run generated (`out/`, Parquet files, the manifest, caches), and changes "
        "the working directory to the copy. The temporary path is never printed, so that "
        "two runs of the notebook give identical output. The folder it started in is kept "
        "as `STARTED_IN`, so that the last cell can step back there and delete the copy.",
        "From here on `Path.cwd()` is a disposable `exercises/`. `SOURCE` still points at "
        "the real one, and is only ever read — for the two files of recorded archive "
        "numbers the slides quote.")
    nb.code(r'''
    def find_exercises():
        here = Path.cwd()
        for candidate in (here, here / "exercises", here.parent / "exercises",
                          here / "Module 2" / "exercises"):
            if (candidate / "lab_support.py").exists() and (candidate / "data" / "make_phones.py").exists():
                return candidate.resolve()
        raise FileNotFoundError("run this notebook from Module 2/notebook or Module 2/exercises")


    STARTED_IN = Path.cwd()
    SOURCE = find_exercises()
    MODULE = SOURCE.parent
    WORK = Path(tempfile.mkdtemp(prefix="module2_notebook_")) / "exercises"
    shutil.copytree(SOURCE, WORK, ignore=shutil.ignore_patterns(
        "out", "*.parquet", "MANIFEST.json", "__pycache__", ".venv", "venv", "landing",
        ".your_attempt", ".attempts.json", ".ipynb_checkpoints", "timing.json"))
    os.chdir(WORK)
    print("working in a temporary copy of Module 2/exercises; the original is only read")
    print("copied:", ", ".join(sorted(p.name for p in WORK.iterdir() if not p.name.startswith("."))))
    ''')

    nb.md("""
    ### The laboratory machinery, in full

    The labs share one support file, `exercises/lab_support.py`: the unsolved marker,
    the loaders, and Module 1's `check_against`, copied there so that two modules
    validate the same table by one rule. Nothing below is imported from the lab
    files: the next cells *are* them.
    """)
    explain(
        "Let the lab files' path constants resolve inside a notebook.",
        "Each file finds its data relative to its own location (`__file__`), which a "
        "notebook does not have.",
        "Points `__file__` at `lab_support.py` in the working copy. Before each later file "
        "is copied in, `__file__` is pointed at that file instead, and set back afterwards.",
        "The next cells can then be copied from the files unchanged.")
    nb.code(r'''
    __file__ = str(Path.cwd() / "lab_support.py")
    assert Path(__file__).exists()
    ''')
    explain(
        "Define the loaders and the upstream check every lab uses.",
        "`load_bus()` returns the real telemetry; `load_phones()` returns generated phone "
        "traces, and says so; `check_against()` is Module 1's declaration run as code — it "
        "asks whether a frame is fit for use by the step that consumes it, which is how Wang "
        "and Strong define data quality [@wang1996].",
        "Copies `lab_support.py` verbatim: the three states a check can report "
        "(`NotSolved`, `EnvironmentNotReady`, or a failure), the loaders, and "
        "`check_against`. Only the file's logging-handler set-up is left out; it matters "
        "only when a generated table is missing, which the next cells prevent.",
        "Every table below comes through one of these functions, exactly as in the labs.")
    nb.source(f"{EX}/lab_support.py", "HERE", "DATA", "BUS_SLICE", "MODULE1_PROFILE", "_log",
              "_warned", "NotSolved", "EnvironmentNotReady", "_fallback_warning", "load_bus",
              "load_phones", "load_simpson", "load_module1_profile", "check_against",
              cite="[@wang1996]")
    repoint("_narrate.py")
    explain(
        "Give the solutions the narration helpers they print with.",
        "Each solution's demonstration tells its story through `narrator`, `show_table` and "
        "`save_figure`; running the demonstration here needs the same three.",
        "Copies `narrator` and `show_table` from `_narrate.py` verbatim (with `__file__` "
        "pointed at that file just before).",
        "The demonstrations further down run unchanged. Their lines begin with the seconds "
        "since the narrator started, which is the one part of the output that differs "
        "from run to run.")
    nb.source(f"{EX}/_narrate.py", "HERE", "OUT", "_START", "_Elapsed", "narrator", "show_table")
    explain(
        "Replace `save_figure` with one that shows the figure in place.",
        "In the terminal `save_figure` writes `out/lab_0K_<name>.html` and a PNG; the "
        "notebook shows figures instead of writing them.",
        "Applies the same layout as `_narrate.py` (template, size, margins), displays the "
        "image, and narrates where it went, as the original does.",
        "This is the only function of the lab machinery that the notebook rewrites.")
    nb.code(r'''
    def save_figure(fig, name, lab, logger=None, width=1000, height=560):
        """The notebook's save_figure: the layout of exercises/_narrate.py, shown in
        place instead of written to out/lab_0K_<name>.html."""
        fig.update_layout(template="plotly_white", width=width, height=height,
                          margin=dict(l=60, r=30, t=60, b=60))
        display(Image(fig.to_image(format="png", width=width, height=height, scale=1)))
        (logger.info if logger else print)(f"figure -> shown here (lab_{lab:02d}_{name})")
    ''')

    nb.md("""
    ### The phone traces: generated, from the archive's measured magnitudes

    The real phone file is personal data, so the labs get a generator instead. Every
    parameter it uses was measured on the real file by `slides/measure.py` and written
    to `data/calibration.json`; the one construction — which volunteer rode which
    shuttle on which day — is labelled as such in the code.
    """)
    repoint("data/make_phones.py")
    explain(
        "Define the generator the labs' phone traces come from.",
        "Block 2 needs the true value of every reading the phones did not record, and "
        "only a generator can have it. Lab 1's Simpson's paradox is planted here too "
        "(`RIDERS`).",
        "Copies `data/make_phones.py` verbatim; `__file__` was pointed at it just before, so "
        "that `calibration.json` is found beside it.",
        "`generate(day, with_truth=True)` returns the student's columns plus the hidden "
        "truth: the signal every reading would have had, the distance to each beacon, and "
        "the aboard state.")
    nb.source(f"{EX}/data/make_phones.py", "HERE", "CALIBRATION", "SEED", "BEACONS", "PROXIMITY",
              "RANGE_METRES", "STRENGTH_AT_ONE_METRE", "PATH_LOSS", "PER_PHONE", "RIDERS",
              "_riders_for", "generate", "simpson_table")
    repoint("data/prepare.py")
    explain(
        "Define the preparation step `setup.sh` runs before any lab.",
        "The labs never generate data at import time; `prepare.py` writes every table "
        "once, verifies the shipped slice against Module 1's declaration, and records a "
        "manifest.",
        "Copies the pieces of `data/prepare.py` verbatim, `main()` included; `__file__` was "
        "pointed at it just before.",
        "Running `main()` in the working copy gives the notebook exactly the files a "
        "student's `bash setup.sh` gives them.")
    nb.source(f"{EX}/data/prepare.py", "HERE", "EXERCISES", "MANIFEST", "SLICE", "MODULE1_PROFILE",
              "PROFILE_SCHEMA", "SIMPSON_COLUMNS", "content_hash", "table_entry",
              "check_module1_profile", "simpson_frame", "main")
    explain(
        "Let the copied code's own `import` lines find the functions defined above.",
        "`prepare.py` imports `generate` from `make_phones` and `check_against` from "
        "`lab_support`; later, `make_figs.py` and the Lab 4 solution import from the other "
        "labs. Importing the files would run the files, not the cells.",
        "`as_module(name, …)` puts a module object holding the notebook's own functions "
        "into `sys.modules`, so `from make_phones import generate` returns the function "
        "defined in the cell above. Then `__file__` is set back to `lab_support.py`.",
        "Every `import` in the copied code resolves to code on this page.")
    nb.code(r'''
    def as_module(name, **objects):
        """A module named `name` whose contents are objects defined in this notebook."""
        module = types.ModuleType(name)
        module.__dict__.update(objects)
        sys.modules[name] = module


    as_module("lab_support", **{name: globals()[name] for name in (
        "NotSolved", "EnvironmentNotReady", "load_bus", "load_phones", "load_simpson",
        "load_module1_profile", "check_against")})
    as_module("make_phones", generate=generate, simpson_table=simpson_table,
              CALIBRATION=CALIBRATION, RIDERS=RIDERS)
    __file__ = str(Path.cwd() / "lab_support.py")
    ''')
    explain(
        "Prepare the data, exactly as `bash setup.sh` does.",
        "The working copy was made without generated files, so they are generated here, "
        "deterministically (seed 20200122).",
        "Calls `prepare.py`'s `main()` when any of the tables the labs read is missing.",
        "Two days of phones, their truth-kept twins, the two-day Simpson frame, and a "
        "manifest of hashes; the slice is verified against Module 1's profile, not "
        "rewritten.")
    nb.code(r'''
    needed = ["phones_2020-01-22.parquet", "phones_truth_2020-01-22.parquet",
              "phones_2020-01-23.parquet", "phones_truth_2020-01-23.parquet", "simpson.parquet"]
    if any(not (Path("data") / name).exists() for name in needed):
        assert main() == 0, "prepare.py failed"
    ''')

    nb.md("### The data, and the archive numbers the slides quote")
    explain(
        "Load the three tables used throughout, and the recorded archive facts.",
        "Most slides compute on the lab data; a few quote the instructor's archive, which "
        "this notebook does not open. Those values were written, as aggregates only, to "
        "`slides/measured.json` (and Module 1's to `Module 1/slides/measured.json`).",
        "Loads the slice, the first generated day as the student sees it, and the same day "
        "with its hidden truth; sorts the slice in time; reads both measured.json files "
        "(read-only, from the repository).",
        "`archive(key)` and `module1(key)` return a recorded value; every use of one is "
        "labelled *archive* in the output.")
    nb.code(r'''
    bus = load_bus()
    phones = load_phones()                       # 2020-01-22, the student's view
    truth = load_phones(with_truth=True)         # the same rows, hidden columns kept
    in_time = (bus.assign(_t=pd.to_datetime(bus["utc_time"], utc=True))
                  .sort_values("_t").reset_index(drop=True))

    FACTS = json.loads((MODULE / "slides" / "measured.json").read_text())
    MODULE1 = json.loads((MODULE.parent / "Module 1" / "slides" / "measured.json").read_text())


    def archive(key):
        """A value measure.py recorded from the archive (slides/measured.json)."""
        return FACTS[key]["value"]


    def module1(key):
        """A value Module 1 recorded from the archive (Module 1/slides/measured.json)."""
        return MODULE1[key]["value"]


    METRES_PER_DEGREE = 111_320    # Module 1's constant for the route's extent


    def to_metres(lat, lon):
        """East and north offsets in metres from the south-west corner of the slice."""
        east = (lon - bus["lon"].min()) * METRES_PER_DEGREE * np.cos(np.radians(bus["lat"].mean()))
        north = (lat - bus["lat"].min()) * METRES_PER_DEGREE
        return np.asarray(east), np.asarray(north)


    print(f"vehicle slice   {len(bus):,} rows x {bus.shape[1]} columns, "
          f"{bus['vehicle_id'].nunique()} vehicle, {in_time['_t'].min():%Y-%m-%d %H:%M} to "
          f"{in_time['_t'].max():%Y-%m-%d %H:%M} UTC")
    print(f"phones, day 1   {len(phones):,} rows x {phones.shape[1]} columns (generated), "
          f"{phones['phone_id'].nunique()} phones")
    print(f"truth, day 1    {truth.shape[1] - phones.shape[1]} hidden columns beside them")
    ''')


# =============================================================================
# Introduction: slides 3-10
# =============================================================================

def introduction() -> None:
    nb.md("""
    ---
    # The case, and the question

    *Slides: "Where this sits — five modules, one case, one thread", "The case — two
    shuttles, sixteen phones, five beacons", "Objectives, prerequisites, and …",
    "… four things people get wrong", "Data cleaning and feature engineering
    overview", "Supporting data products on layered data architecture", "IID vs
    timeseries" and "Two sources observe one behaviour from opposite sides, and
    neither answers alone".*

    One case runs through the course: automated shuttles on a fixed loop in
    Copenhagen in January 2020, and volunteers carrying instrumented phones. The
    question is whether a given volunteer was on a given shuttle at a given moment.
    The vehicle file observes the shuttle; the phone file observes the passenger;
    neither answers alone, and this module is the work of making them one table.

    The six objectives, in one line each: join the two files on shared time; account
    for every row; say whether a two-day difference is the days or the fleet; measure
    what a fill invents; fit every constant on the training rows only; and find the
    column that already contains the answer.

    *Not redrawn:* the title photograph and logo; the video on "Where this sits" and
    its poster frame; two generated posters ("Ingesting trust", "Four data engineering
    misconceptions"), whose two numbers — 13.6 decibels invented and 64 per cent of
    the variation destroyed — are recomputed in Block 2; and the bronze, silver and
    gold pipeline diagram, a vendor drawing of layers that carries no data.
    """)

    explain(
        "Draw the route the shuttle drove, from its own positions.",
        "The slide shows a schematic of the loop and its three stops. The slice holds every "
        "position the vehicle reported, and the doors' state, so the loop and the places "
        "where the doors opened can be drawn from the data itself.",
        "Converts latitude and longitude to metres with Module 1's constant "
        "(111,320 m per degree, the east axis scaled by the cosine of the latitude), plots "
        "every reading, marks the readings taken with the doors open, and measures the "
        "bounding box. Then finds the stops: each unbroken run of \"opened\" readings in time "
        "order is one door opening, placed at its median position; openings closer than "
        "10 metres are joined (single linkage), and a place where the doors opened at least "
        "five times is called a stop. The stops are lettered A, B, C by how often the doors "
        "opened there. That is an inference from the doors: the slice does not name its "
        "stops, so these letters need not be the archive's \"Stop A\", \"Stop B\", \"Stop C\".",
        "The whole trial fits in a box of about 67 by 86 metres, and the doors opened at "
        "three places — the slide's three stops, found from the data. Beacon range is tens "
        "of metres, which is why, in Block 2, hearing the vehicle's beacon cannot tell aboard "
        "from waiting at a stop.")
    nb.figure("route_map", r'''
    from scipy.cluster.hierarchy import fcluster, linkage

    east, north = to_metres(bus["lat"], bus["lon"])
    doors_open = (bus["door_state"] == "opened").to_numpy()
    fig = go.Figure()
    fig.add_scatter(x=east[~doors_open], y=north[~doors_open], mode="markers",
                    marker=dict(color=BLUE, size=3, opacity=0.25), name="position, doors closed")
    fig.add_scatter(x=east[doors_open], y=north[doors_open], mode="markers",
                    marker=dict(color=ORANGE, size=4, opacity=0.5), name="position, doors opened")
    width_m, height_m = float(east.max()), float(north.max())
    fig.add_shape(type="rect", x0=0, y0=0, x1=width_m, y1=height_m,
                  line=dict(color=GREY, dash="dash"))

    # Stops, inferred: one door opening = one unbroken run of "opened" in time order.
    opened_in_time = (in_time["door_state"] == "opened").to_numpy()
    run = np.r_[0, np.cumsum(opened_in_time[1:] != opened_in_time[:-1])]
    run_east, run_north = to_metres(in_time["lat"], in_time["lon"])
    openings = (pd.DataFrame({"run": run, "east": run_east, "north": run_north})[opened_in_time]
                  .groupby("run")[["east", "north"]].median())
    place = fcluster(linkage(openings.to_numpy(), "single"), 10, "distance")
    places = (openings.assign(place=place).groupby("place")
                .agg(east=("east", "mean"), north=("north", "mean"), openings=("east", "size")))
    stops = (places[places["openings"] >= 5].sort_values("openings", ascending=False)
               .reset_index(drop=True))
    stops.index = [f"stop {letter}" for letter in "ABCDEFG"[:len(stops)]]
    fig.add_scatter(x=stops["east"], y=stops["north"], mode="markers+text", text=list(stops.index),
                    textposition="middle right", textfont=dict(size=15, color=NAVY),
                    marker=dict(color=NAVY, size=16, symbol="diamond-open", line=dict(width=3)),
                    name="stop, inferred from where the doors opened (letters are the notebook's)")
    fig.update_layout(title=f"The loop from the vehicle's own positions: "
                            f"{height_m:.0f} m north-south by {width_m:.0f} m east-west",
                      xaxis_title="metres east", yaxis_title="metres north",
                      yaxis_scaleanchor="x", legend=dict(orientation="h", y=-0.15))
    show(fig, "route_map", width=800, height=720)
    agrees("route extent north-south, metres (Module 1's recipe)", height_m, module1("extent_m")[0], 0)
    agrees("route extent east-west, metres", width_m, module1("extent_m")[1], 0)
    beside("vehicle file", f"{len(bus):,} rows, {bus.shape[1]} columns, {bus['vehicle_id'].nunique()} vehicle",
           f"{archive('bus_rows'):,} rows, {module1('columns')} columns, {module1('vehicles')} vehicles (archive)",
           "the slide counts the archive's bus file, both shuttles; the lab slice keeps the one that ran on both days")
    print(f"  door openings: {len(openings)}; places where the doors opened: {len(places)}; "
          f"with at least five openings (stops): {len(stops)}")
    print(stops.round(1).to_string())
    ''', slides=["4"], treatment="lab data: the slide's schematic replaced by the loop drawn "
                                "from the slice's positions; stops A, B, C inferred from where "
                                "the doors opened (the letters are the notebook's)")

    explain(
        "Show why a time series is not a bag of independent draws.",
        "The slide contrasts independent draws with an ordered series. On this telemetry "
        "the difference is a number: consecutive speeds correlate at 0.9967 on 22 January "
        "(Module 1's measurement); the same values shuffled are measured below. A series "
        "that correlates with itself holds fewer independent observations than it has "
        "readings, which Block 1 quantifies as the effective sample size [@bayley1946].",
        "Takes ten minutes of 22 January's speed from 10:15 UTC in time order and the same values shuffled (seed "
        "20200122), draws both, and plots each reading against the one before it. The "
        "autocorrelation is the Box–Jenkins estimator the course grades, one mean and one "
        "sum of squares [@box2015].",
        "Order carries information. A random split, a z-score over readings and a plain "
        "standard error all assume it does not.")
    nb.figure("iid_vs_series", r'''
    def lag_one(values):
        x = np.asarray(values, float)
        centred = x - x.mean()
        return float(np.sum(centred[1:] * centred[:-1]) / np.sum(centred ** 2))

    slice_day_one = in_time[in_time["_t"].dt.date.astype(str) == "2020-01-22"]
    stretch = slice_day_one[(slice_day_one["_t"] >= "2020-01-22 10:15:00+00:00")
                            & (slice_day_one["_t"] < "2020-01-22 10:25:00+00:00")]["speed"].to_numpy()
    shuffled = np.random.default_rng(20200122).permutation(stretch)
    fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3], vertical_spacing=0.16,
                        subplot_titles=("in time order",
                                        f"each reading against the one before: r₁ = {lag_one(stretch):.3f}",
                                        "the same values, shuffled",
                                        f"each against the one before: r₁ = {lag_one(shuffled):.3f}"))
    for row, values, colour in ((1, stretch, BLUE), (2, shuffled, GREY)):
        fig.add_scatter(y=values, mode="lines", line=dict(color=colour, width=1),
                        showlegend=False, row=row, col=1)
        fig.add_scatter(x=values[:-1], y=values[1:], mode="markers",
                        marker=dict(color=colour, size=3, opacity=0.4), showlegend=False,
                        row=row, col=2)
    fig.update_annotations(font_size=12)
    fig.update_xaxes(title_text="reading (0.5 s apart)", row=2, col=1)
    fig.update_yaxes(title_text="speed, m/s", row=1, col=1)
    fig.update_yaxes(title_text="speed, m/s", row=2, col=1)
    fig.update_layout(title="A time series (top) and the same numbers as independent draws (bottom)")
    show(fig, "iid_vs_series", height=600)
    agrees("lag-1 autocorrelation of speed, 22 January, in time order",
           lag_one(slice_day_one["speed"]), module1("speed_autocorrelation_lag1"), 4)
    print(f"  the ten minutes drawn: r1 = {lag_one(stretch):.4f} in order, "
          f"{lag_one(shuffled):.4f} shuffled")
    ''', slides=["9"], treatment="lab data: the slide's icon drawing replaced by ten minutes "
                                "of the slice's speed, in order and shuffled")

    explain(
        "Put the two clocks on one axis.",
        "The alignment slides say the vehicle reports every 0.5 seconds and the phones "
        "every 0.993, so no reading in one file has a partner in the other. Drawing six "
        "seconds of both makes the grain choice visible.",
        "Measures both median intervals (per vehicle and per phone, within a day), then "
        "draws the vehicle's readings and one phone's readings between 09:00:00 and "
        "09:00:06 UTC on 22 January, with the five-second tumbling windows behind them.",
        "The phone's ticks drift against the vehicle's by 0.007 s a reading; a window, not a "
        "timestamp, is what the two can share.")
    nb.figure("two_clocks", r'''
    vehicle_step = float(in_time.groupby(in_time["_t"].dt.date)["_t"].diff().dt.total_seconds().median())
    phone_times = pd.to_datetime(phones["timestamp_utc"], utc=True)
    phone_step = float(phones.assign(_t=phone_times).sort_values(["phone_id", "_t"])
                       .groupby("phone_id")["_t"].diff().dt.total_seconds().median())
    agrees("vehicle interval, seconds", vehicle_step, archive("bus_interval_s"), 3)
    agrees("phone interval, seconds (generated at the archive's rate)", phone_step,
           archive("phone_interval_s"), 3)

    start = pd.Timestamp("2020-01-22 09:00:00", tz="UTC")
    stop = start + pd.Timedelta(seconds=6)
    v = in_time[(in_time["_t"] >= start) & (in_time["_t"] < stop)]["_t"]
    p = phone_times[(phones["phone_id"] == "p00") & (phone_times >= start) & (phone_times < stop)]
    seconds = lambda stamps: (stamps - start).dt.total_seconds()
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=5, fillcolor="rgba(42,120,214,0.06)", line_width=0,
                  annotation_text="window [09:00:00, 09:00:05)", annotation_position="top left")
    fig.add_vline(x=5, line=dict(color=GREY, dash="dash"))
    fig.add_scatter(x=seconds(v), y=np.ones(len(v)), mode="markers",
                    marker=dict(symbol="line-ns-open", size=26, color=BLUE, line=dict(width=2)),
                    name=f"vehicle, median every {vehicle_step:.3f} s")
    fig.add_scatter(x=seconds(p), y=np.zeros(len(p)), mode="markers+text",
                    marker=dict(symbol="line-ns-open", size=26, color=ORANGE, line=dict(width=2)),
                    text=[f"{s:.3f}" for s in seconds(p)], textposition="bottom center",
                    name=f"phone p00, every {phone_step:.3f} s")
    fig.update_yaxes(tickvals=[0, 1], ticktext=["phone", "vehicle"], range=[-0.8, 1.6])
    fig.update_layout(title="Two clocks, no shared timestamp: six seconds of 22 January",
                      xaxis_title="seconds after 09:00:00 UTC", legend=dict(orientation="h", y=-0.25))
    show(fig, "two_clocks", height=380)
    print("  phone ticks, seconds:", ", ".join(f"{s:.3f}" for s in seconds(p)))
    beside("the fifth phone tick on the slide's drawing", f"{4 * phone_step:.3f} s", "2.973 s",
           "the generated poster labels the fifth tick 2.973 s; four steps of 0.993 s are 3.972 s")
    ''', slides=["10", "22"], treatment="lab data: the slide's generated drawing of the two "
                                        "clocks replaced by the slice's and the phones' own "
                                        "timestamps")

    explain(
        "Set the two sources side by side, and the labels they carry.",
        "The slide describes the archive: 53,155 vehicle rows, 40 phone columns from 16 "
        "volunteers, labels on 100 per cent of the first day and none of the second. The "
        "lab data differ in known ways, and the differences matter later (Module 5 starts "
        "from the missing second-day labels).",
        "Counts the same quantities on the slice and the two generated days, and prints the "
        "archive's recorded values beside them.",
        "The generator labels both days — it has to, to plant Lab 1's reversal — so "
        "\"one day of labels\" is a fact about the archive that the lab data do not "
        "reproduce.")
    nb.code(r'''
    day_two = load_phones(day="2020-01-23")
    coverage = {day: round(float(frame["label2"].notna().mean()) * 100, 1)
                for day, frame in (("2020-01-22", phones), ("2020-01-23", day_two))}
    beside("phone file", f"{phones.shape[1]} columns (generated), "
           f"{phones['phone_id'].nunique()} + {day_two['phone_id'].nunique()} phones on the two days",
           f"{archive('phone_columns')} columns, {archive('phones')} volunteers "
           f"({archive('phones_per_day')['2020-01-22']} + {archive('phones_per_day')['2020-01-23']}) (archive)",
           "the generator keeps the per-day phone counts and only the columns the labs use")
    beside("label coverage by day, per cent", coverage, f"{archive('label_coverage_per_day')} (archive)",
           "the generator labels both days so that Lab 1's two-day comparison exists; in the "
           "archive only 22 January was labelled")
    agrees("aboard share of labelled rows, first day, per cent",
           100 * (phones["label2"] == "IN").mean(), archive("aboard_share_of_labelled"), 1)
    ''')


# =============================================================================

def coalesce_streams(path: Path) -> None:
    """Join consecutive pieces of one stream in each cell's stored output.

    The kernel may deliver a cell's printed text in one piece or in several, which
    makes two identical runs store different output lists. A local helper, so that
    tools/notebook_kit.py stays as it is."""
    import nbformat
    notebook = nbformat.read(str(path), as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        joined = []
        for output in cell.get("outputs", []):
            if (output.get("output_type") == "stream" and joined
                    and joined[-1].get("output_type") == "stream" and joined[-1]["name"] == output["name"]):
                joined[-1]["text"] += output["text"]
            else:
                joined.append(output)
        cell["outputs"] = joined
    nbformat.write(notebook, str(path))


def main() -> int:
    front_matter()
    setup()
    introduction()
    from sections import block1, block2, block3, block4, closing
    block1.build(nb, explain)
    block2.build(nb, explain)
    block3.build(nb, explain)
    block4.build(nb, explain)
    closing.build(nb, explain)
    nb.md("""
    ---
    ## Where this notebook and the slides differ, and why

    The slides are the master and are not changed. Where a number computed here does
    not match the number a slide prints, the notebook printed both at that point, with
    the reason. The table gathers every one of them, as they were printed in this run.

    Differences in wording and labels, which the table cannot hold:

    - **Slide 20** (the normal-distribution figure): the density formula lacks σ under
      the root and its exponent is garbled; the axis ticks read μ−3σ, μ−σ, μ, μ+σ, μ+2σ,
      which is not symmetric. The notebook draws the density from `scipy.stats.norm`.
    - **Slide 32** (the MCAR/MAR/MNAR picture): "the vehicle beacon is silent on 57 % to
      86 %" is the range over all five beacons; the vehicle beacon alone is 57.0 % on
      the archive.
    - **Slide 42**: "preserves 87 % of the original variance" — 0.87 is a ratio of
      standard deviations, so the variance kept is 0.87² ≈ 0.76; likewise 0.36 keeps
      0.13 of the variance.
    - **Source lines** on several slides credit `make_figs.py` for archive numbers that
      `slides/measure.py` produces; `make_figs.py` never opens the archive.
    - **Slide 67** spells "diffierent" and "Whatch".
    - **Slide 4** (the case): the stops on the route map are lettered A, B, C by the
      notebook, from where the vehicle's doors opened; the slice does not name its stops,
      so the letters need not match the archive's "Stop A", "Stop B", "Stop C".
    - **Slide 33** (two causes of absence): the slide's diagram draws the phone against
      two shuttles; the lab slice holds one, so the notebook's redrawing measures the
      phone's second distance to a stop, and places the phone by hand.
    - **Slide 39** and the Lab 2 files: the masked moving average is the causal recursion
      s_t = α·x_t + (1 − α)·s_{t−1}; Servizi et al. (2023), Appendix B, to which the lab's
      docstrings attribute it, describe a window average weighted towards the window's
      centre. Same family, different filter; the lab's docstrings are quoted unchanged.
    - **Slide 83**: "costs accuracy on all of them" — a nearest match never changes the
      partner of a row that was already matched, so widening the tolerance does not make
      those rows worse; it removes the guarantee that any matched row is within one
      second. Measured beside the slide's figure, with the accuracy the added rows lose.
    - **Slides 81 and 85, and Labs 3 and 4**: the split by row puts one instant on both
      sides (eleven rows train, one tests), and the Lab 4 solution fits the hand-off's
      transform on three rows that `assemble` then sends to test. Both are in the lab
      files, which are delivered and not changed; both are counted in the table, and
      the notebook's own hand-off is fitted on the training rows.
    - **Citations inside the lab stubs**, quoted verbatim and not changed: Wang and
      Strong (1996) is cited for the conservation ledger, which the paper does not state
      (it defines data quality as fitness for use, which fits the profile check); McKinney
      (2022) is cited for the nearest match, whose `merge_asof` the book does not cover.
    - **Lab 3's typed evidence**: `phone_speed`'s purity, cardinality and gap and the gaps
      of the row counter and `rssi1` are typed into the solution; the notebook recomputes
      all five with a stated recipe and they agree.
    - **The stubs' slide titles** are checked against the deck in the cell below; each
      one that is not a shown slide's title is a row of the table.
    """)
    explain(
        "Check every slide title the lab stubs quote against the deck's own titles.",
        "The stubs send the student to slides by title. A title that is not in the deck, or "
        "is on a hidden slide, sends them nowhere; the slides are not changed, so the "
        "notebook says where each one actually is.",
        "Reads the slide titles and the hidden flags from `slides/Module2.pptx` (the file's "
        "XML, read-only); takes from each stub's docstrings the block title after \"Where "
        "it sits\" and every quoted \"Definition — …\"; looks each up among the titles. A "
        "title found on a shown slide is printed with its number; any other is printed "
        "beside the deck's slide that makes its point — a mapping the notebook states, "
        "checked to be a real title — and recorded for the table.",
        "Every quoted title either names a shown slide or has its row in the table below.")
    nb.code(r'''
    import ast
    import re
    import zipfile
    from xml.etree import ElementTree

    DRAWING = "http://schemas.openxmlformats.org/drawingml/2006/main"
    PRESENTATION = "http://schemas.openxmlformats.org/presentationml/2006/main"
    RELATION = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    tidy = lambda text: re.sub(r"\s+", " ", text).strip()


    def deck_titles(path):
        """{title: (slide number, hidden)} for every slide of a .pptx, read from its XML."""
        titles = {}
        with zipfile.ZipFile(path) as deck:
            presentation = ElementTree.fromstring(deck.read("ppt/presentation.xml"))
            targets = {r.get("Id"): r.get("Target")
                       for r in ElementTree.fromstring(deck.read("ppt/_rels/presentation.xml.rels"))}
            for number, slide_id in enumerate(presentation.find(f"{{{PRESENTATION}}}sldIdLst"), start=1):
                slide = ElementTree.fromstring(deck.read("ppt/" + targets[slide_id.get(f"{{{RELATION}}}id")]))
                for shape in slide.iter(f"{{{PRESENTATION}}}sp"):
                    placeholder = shape.find(f".//{{{PRESENTATION}}}nvPr/{{{PRESENTATION}}}ph")
                    if placeholder is not None and placeholder.get("type") in ("title", "ctrTitle"):
                        text = " ".join("".join(t.text or "" for t in paragraph.iter(f"{{{DRAWING}}}t"))
                                        for paragraph in shape.iter(f"{{{DRAWING}}}p"))
                        titles[tidy(text)] = (number, slide.get("show") == "0")
        return titles


    def quoted_titles(stub):
        """The block title and every "Definition — …" a stub's docstrings quote."""
        tree = ast.parse(stub.read_text(encoding="utf-8"))
        text = tidy(" ".join(ast.get_docstring(node) or "" for node in [tree] + [
            node for node in tree.body if isinstance(node, ast.FunctionDef)]))
        found = re.findall(r'Where it sits: Block \w+ — "([^"]+)"', text) + re.findall(r'"(Definition — [^"]+)"', text)
        return list(dict.fromkeys(found))


    titles = deck_titles(MODULE / "slides" / "Module2.pptx")
    # The notebook's reading of which slide makes each missing title's point.
    CLOSEST = {
        "Bronze to silver — what changes, and what must not": "Silver is derived and rebuildable, and bronze must never be modified",
        "The mask, and why it must survive": "A fill must be recorded, because a filled value and a measured value look alike",
        "The leak in this archive": "BusID is not correlated with the target, it is the target under another name",
        "What the join costs": "Widening the tolerance thirtyfold buys few rows and costs accuracy on all of them",
        "Definition — the six window features": "Definition — six window features",
        "Definition — a split by time, never at random": "Definition — a split by time, never at random for timeseries",
        "Definition — a nearest match within a tolerance": "Definition — a nearest match within a tolerance (fuzzy time-based join)",
        "Definition — the table this module hands to the next three": "Definition — the table this module hands to the next steps",
    }
    for stub in sorted((Path.cwd() / "labs").glob("0*.py")):
        for title in quoted_titles(stub):
            if title in titles and not titles[title][1]:
                print(f"  {stub.name}: \"{title}\" is slide {titles[title][0]}")
            elif title in titles:
                beside(f"{stub.name}: slide title", f"slide {titles[title][0]}, hidden in the deck",
                       f"\"{title}\" (the stub)", "the slide exists but is not shown in the lecture; "
                       "the stub states the definition in full")
            else:
                deck_title = CLOSEST[title]
                number, hidden = titles[deck_title]           # a KeyError here would mean a wrong mapping
                assert not hidden
                beside(f"{stub.name}: slide title", f"slide {number}, \"{deck_title}\"",
                       f"\"{title}\" (the stub)", "no slide has the stub's title; this is the slide that makes its point")
    ''')
    explain(
        "Gather every number this notebook printed beside a slide's number.",
        "A reader should find every difference between the notebook and the slides in one "
        "place, with its reason, without searching the notebook.",
        "Tabulates what each `beside()` call recorded during this run: the quantity, the value "
        "computed here, the value on the slide or in the archive, and why they differ.",
        "Every row is explained where it first appears; none of them is a change to the slides.")
    nb.code('''
    with pd.option_context("display.max_colwidth", None):     # the reasons, in full
        display(pd.DataFrame(DIFFERENCES, columns=["quantity", "computed here",
                                                   "the slide or the archive", "why they differ"]))
    ''')
    explain(
        "Delete the temporary copy of the exercises.",
        "The notebook worked in a copy so as never to write into the student's folder; the "
        "copy, with the Parquet files, the hand-off and the figures' data it generated, "
        "should not outlive the run.",
        "Steps back to the folder the notebook started in (`STARTED_IN`), removes the "
        "temporary folder with everything in it, and confirms that it is gone. The "
        "student's `exercises/` was only ever read.",
        "Running the notebook leaves nothing behind but its own outputs.")
    nb.code(r'''
    os.chdir(STARTED_IN)
    shutil.rmtree(WORK.parent)
    print("temporary copy of exercises/ removed:", not WORK.parent.exists())
    ''')
    nb.write(OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    if "--no-run" not in sys.argv:
        execute(OUTPUT, HERE)
        coalesce_streams(OUTPUT)
        print(f"executed {OUTPUT.relative_to(ROOT)} from {HERE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
