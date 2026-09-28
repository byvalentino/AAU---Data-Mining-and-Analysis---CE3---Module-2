"""The closing slides — bias, the regulation, the hand-off, the capabilities — and
the practice questions with their answers."""


def build(nb, explain) -> None:
    nb.md("""
    ---
    # What the module measured, and what it hands on

    *Slides: "Four of Module 1's ten flavours of bias were measured in this module",
    "Article 10 requires an examination for bias, and this module performed one", "The
    hand-off is four objects, built by Lab 4 and graded by its check" and "What you can
    now do".*
    """)
    explain(
        "Recompute the numbers the bias slide gathers, and label the archive's.",
        "The slide ties four of Module 1's ten flavours of bias to this module's "
        "measurements: measurement and labelling, missing-data related, leakage and "
        "peeking, and time-related.",
        "Prints each number from the cell that measured it above, or from the archive's "
        "record where only the archive has it.",
        "Every number on the slide has a line here; the archive's are labelled as such.")
    nb.code(r'''
    beside("beacons silent, per cent of rows", f"{min(generated_absence.values())} to {max(generated_absence.values())} (generated)",
           "57 to 86.3 (archive)", "the generator matches the archive's heard rates per aboard state")
    agrees("mean fill: decibels invented", biases["mean"], 13.6, 1)
    agrees("mean fill: spread kept", ratios["mean"], 0.36, 2)
    beside("BusID empty, per cent of rows", round(100 * float(phones["bus_id"].isna().mean()), 2),
           f"{archive('busid_absent_share')} (archive)", "the archive's share covers its unlabelled second day")
    agrees("lag-1 autocorrelation of speed, 22 January", lag_one(slice_day_one["speed"]), 0.9967, 4)
    ''')
    nb.md("""
    ### What the law asks

    The Artificial Intelligence Act [@aiact2024], Article 10:

    - **10(2)(f)** — training, validation and testing data are examined in view of
      possible biases.
    - **10(2)(g)** — appropriate measures detect, prevent and mitigate the biases the
      examination finds.
    - **10(5)** — special categories of personal data may be processed for bias
      detection and correction only where strictly necessary, and under safeguards.

    Simpson's paradox in Block 1 is such an examination, and it makes the slide's
    point by itself: the pooled comparison and the profile check on the pool both
    report nothing, and the per-group ones do not. The deck adds that the Digital
    Omnibus sets 2 December 2027 for the Annex III high-risk obligations
    [@omnibus2026]. *This is the text of the Regulation, not legal advice.*

    ### The hand-off, and what you can now do

    The table (one row per phone per window, masks beside the filled values, the
    target), the split recorded as an instant, the fitted transform, and the ledger —
    four objects, one manifest, written by Lab 4 to `out/handoff/`. One fact is carried
    forward on purpose: in the archive, labels exist for the first day and not the
    second, and Module 5 begins there.

    Put two sources on one grain and prove no row went missing unnoticed; ask why a
    value is absent before filling it, and report what the fill invented and what it
    flattened; fit a transform on the training rows and apply it unchanged; recognise a
    column that already contains the answer; build window features, split by time, and
    measure what a join costs instead of estimating it.
    """)

    explain(
        "Record the deck's source lines that name the wrong script.",
        "Eleven shown slides carry the source line \"Numbers: the archive — "
        "data/passengers.csv and data/bus.csv … Measured by Module 2/slides/make_figs.py\" "
        "(the two alignment slides, the distribution and theorem slides, the absence, beacon "
        "and BusID slides, the window-feature and split slides, the join slide and the bias "
        "summary). The convolution slide's says \"Numbers: the course generator — "
        "make_phones.py\". The scripts can be read.",
        "Reads `slides/make_figs.py` and `slides/measure.py` from the repository and asks "
        "which of them opens a CSV file, and whether the convolution figure's function "
        "touches the generator.",
        "`measure.py` measures the archive; `make_figs.py` never opens it — it reads "
        "`measured.json` and the generator — and the convolution figure uses no data at "
        "all, as its own caption says. The footers should name `measure.py`, and the "
        "convolution slide's should say no data.")
    nb.code(r'''
    figs_code = (MODULE / "slides" / "make_figs.py").read_text()
    measure_code = (MODULE / "slides" / "measure.py").read_text()
    convolution_code = figs_code.split("def figure_convolution")[1].split("\ndef ")[0]
    beside("script that measured the archive numbers",
           f"make_figs.py opens a CSV: {'read_csv' in figs_code}; measure.py opens a CSV: {'read_csv' in measure_code}",
           "\"Measured by Module 2/slides/make_figs.py\" (slides 10, 13, 14, 22, 34, 35, 56, 66, 81, 83, 88)",
           "make_figs.py reads measured.json and the generator; the archive is measured by measure.py")
    beside("data behind the convolution figure",
           f"figure_convolution calls the generator or a loader: {any(call in convolution_code for call in ('generate(', 'load_phones(', 'read_csv('))}",
           "\"Numbers: the course generator — make_phones.py\" (slide 77)",
           "the figure draws the theorem from closed forms; its own caption on the slide says it uses no data")
    ''')

    nb.md("""
    ---
    ## Practice

    Four questions, each a few lines, each with a definite answer. Answers follow.

    1. **Can any beacon beat the base rate by its value?** Block 2 showed that "heard"
       does not. Among the rows where a beacon *was* heard, does the signal strength
       separate aboard from not aboard?
    2. **What does the grain cost?** Count the distinct 1-second, 5-second and
       30-second windows the first day's phone rows fall into. What is thrown away at
       each step?
    3. **Is `label` safe to use as a feature?** `label2` is the target. Cross-tabulate
       and decide, and say what class of leak it is.
    4. **Which comparison did you make?** The aboard share moves one way on each shuttle
       and the other way pooled. Which should a transport authority publish, which should
       an engineer act on, and what would settle the disagreement?
    """)
    explain(
        "Answer question 1: compare the signal strength of heard readings, aboard against "
        "not aboard.",
        "A column whose presence carries nothing may still carry something in its value.",
        "For each beacon, among the rows where it was heard on the generated first day: the "
        "mean signal strength aboard and not aboard, the spread of the heard readings, and "
        "the area under the ROC curve of the signal strength as a score for \"aboard\" "
        "(0.5 is a coin; 1 separates perfectly).",
        "Four beacons give areas between about 0.4 and 0.6 — little better than a coin. "
        "`rssi2` is the exception: 0.26, or 0.74 read the other way, because in the generated "
        "day its heard readings are weaker when the passenger is aboard. A value can carry "
        "what presence does not; but the generator was calibrated on the archive's presence "
        "rates, not its signal values, so this is a lead to test on the archive, not a "
        "finding.")
    nb.code(r'''
    from sklearn.metrics import roc_auc_score
    for b in ["rssiA", "rssiB", "rssiC", "rssi1", "rssi2"]:
        heard_rows = labelled[labelled[b].notna()]
        on = heard_rows.loc[heard_rows["label2"] == "IN", b].mean()
        off = heard_rows.loc[heard_rows["label2"] == "OUT", b].mean()
        area = roc_auc_score(heard_rows["label2"] == "IN", heard_rows[b])
        print(f"  {b:7} aboard {on:7.1f} dBm   not aboard {off:7.1f} dBm   difference {on - off:+5.1f}"
              f"   spread {heard_rows[b].std(ddof=0):4.1f}   area under the ROC curve {area:.2f}")
    ''')
    explain(
        "Answer question 2: count the windows at three grains.",
        "Every grain is a choice, and the ledger records it because every later number "
        "depends on it.",
        "Floors the first day's phone timestamps (UTC) to 1, 5 and 30 seconds and counts "
        "distinct windows and phone rows per window.",
        "At one second a window holds about one reading per phone; at thirty seconds, about "
        "thirty, and everything that happened inside them is averaged away. Coarser grains "
        "keep more rows per window and less of what happened.")
    nb.code(r'''
    stamps = pd.to_datetime(phones["timestamp_utc"], utc=True)
    for grain in ("1s", "5s", "30s"):
        count = stamps.dt.floor(grain).nunique()
        print(f"  {grain:>4}: {count:6,} windows, {len(stamps) / count:6.1f} phone rows per window")
    ''')
    explain(
        "Answer question 3: cross-tabulate `label` against the target.",
        "A column written by the same process that writes the target is the target under "
        "another name [@kaufman2012].",
        "Cross-tabulates the generated first day's `label` against `label2`, and asks the "
        "Lab 3 detector.",
        "Every value of `label` maps to exactly one value of the target: it is the target's "
        "own source, the same class of leak as `bus_id`, and `find_leaks` names it.")
    nb.code(r'''
    print(pd.crosstab(phones["label"], phones["label2"]).to_string())
    print("\nfind_leaks names label:", "label" in find_leaks(phones, TARGET))
    ''')
    nb.md("""
    **Answer to question 4.** Both numbers are right, and they answer different
    questions. The pooled share is what the fleet delivered on each day, which is what a
    transport authority is accountable for. The per-shuttle share is what a shuttle does
    to the people on it, which is what an engineer changes. What settles the
    disagreement is knowing whether the day caused the fleet mix to change: if it did,
    the pooled comparison is the effect of the day; if the mix changed for an unrelated
    reason, the pooled comparison is an artefact of composition [@pearl2014].

    **The module's exam question** (from `exercises/READING.md`): *you are given a table
    someone else cleaned; it has no missing values — what do you ask, and why? Then
    describe one defect that would survive every question you just asked.* Every block
    of this notebook is part of an answer: the ledger and the masks for the first half,
    leakage and the pooled-versus-grouped comparison for the second.

    *Every output above is the author's own, computed by this notebook from
    `exercises/data/bus_slice.csv.gz` and from the course's generator
    (`exercises/data/make_phones.py`, seed 20200122); numbers labelled "archive" are read
    from `slides/measured.json` and `Module 1/slides/measured.json`, where
    `slides/measure.py` and Module 1's scripts recorded them from the archive, in
    aggregate only.*
    """)
