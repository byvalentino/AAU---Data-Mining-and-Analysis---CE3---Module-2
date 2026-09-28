"""Block 3 — features, and the transform that is part of the model — and
Laboratory 3: encoding, target encoding, scaling, what a window summary hides,
the fitted transform, the leak rule, the leak in this archive, the verdict, and
the Lab 3 solution."""

EX = "Module 2/exercises"
S = f"{EX}/solutions"
LABS = f"{EX}/labs"
FIGS = "Module 2/slides/make_figs.py"


def build(nb, explain) -> None:
    nb.md("""
    ---
    # Block 3 — Features, and the transform that is part of the model

    **The question of this block:** which constants does a pipeline learn, and which
    column already knows the answer?

    *Slide: "A feature encodes a claim about the problem, and the model inherits it".*
    Each feature encodes a belief — speed averaged over five seconds asserts that the
    average matters more than the instant — so features are written down and reviewed,
    not accumulated. The slide names one cheap feature: whether the vehicle beacon was
    heard at all, a boolean made from an absence. Block 2 already measured it: heard on
    8.8 per cent of aboard rows and 14.4 per cent of not-aboard rows, below the base
    rate. It is cheap, and it carries almost nothing here.
    """)

    # --- slides 47-48: encoding ---------------------------------------------------------
    nb.md("### Encoding categories\n\n*Slides: \"Encoding categorical variables\" and \"Target "
          "encoding writes the target into a feature, and usually leaks\".*")
    explain(
        "One-hot encode the phones' `label` column.",
        "A model needs numbers. Label encoding (1, 2, 3) invents an order; one-hot gives "
        "each category its own column and states none [@mckinney2022, § 7.2].",
        "Takes two rows of each of the three labels the generated phones carry and "
        "encodes them with `pandas.get_dummies`, drawing the result as a table.",
        "Three categories become three columns of 0 and 1; exactly one is 1 in each row.")
    nb.figure("one_hot", r'''
    sample = pd.concat([phones[phones["label"] == value].head(2) for value in sorted(phones["label"].unique())])
    encoded = pd.get_dummies(sample["label"]).astype(int)
    columns = ["label"] + list(encoded.columns)
    cells = [list(sample["label"])] + [list(encoded[c]) for c in encoded.columns]
    fig = go.Figure(go.Table(
        header=dict(values=columns, fill_color=[GREY] + [BLUE] * len(encoded.columns),
                    font=dict(color="white", size=16), height=36),
        cells=dict(values=cells, font=dict(size=15), height=32,
                   fill_color=[["#F2F2F0"] * len(sample)] + [["#EAF2FB"] * len(sample)] * len(encoded.columns))))
    fig.update_layout(title="One-hot encoding of the phones' label (generated, first day)",
                      margin=dict(l=30, r=30, t=70, b=20))
    show(fig, "one_hot", height=380)
    print("  one 1 per row:", bool((encoded.sum(axis=1) == 1).all()))
    ''', slides=["47"], treatment="lab data: the slide's vendor graphic replaced by the generated "
                                 "phones' own label column, encoded")
    explain(
        "Measure the leak in target encoding computed over the whole table.",
        "Target encoding replaces a category with the average target for that category "
        "[@micci2001]. Computed over all rows, it tells every test row about its own label "
        "[@kaufman2012; @kapoor2023].",
        "Sorts the first generated day by time and splits it 70/30. Encodes `phone_id` by "
        "each phone's aboard share, once over all rows and once over the training rows "
        "only, and scores each encoding as a probability on the test rows (the Brier score, "
        "lower is better).",
        "The whole-table encoding scores better on the test rows because it has read them. "
        "Computed inside the training fold, the encoding is honest; the same holds for any "
        "scaler.")
    nb.code(r'''
    ordered = phones.sort_values("timestamp_utc").reset_index(drop=True)
    cut = int(len(ordered) * 0.7)
    is_aboard = (ordered["label2"] == "IN").astype(float)
    everywhere = is_aboard.groupby(ordered["phone_id"]).mean()
    training_only = is_aboard[:cut].groupby(ordered["phone_id"][:cut]).mean()
    test_phone, test_truth = ordered["phone_id"][cut:], is_aboard[cut:]
    brier = {name: float(((test_phone.map(encoding) - test_truth) ** 2).mean())
             for name, encoding in (("computed over all rows (leaks)", everywhere),
                                    ("computed on the training rows only", training_only))}
    for name, score in brier.items():
        print(f"  target encoding {name:<36} Brier score on the test rows {score:.4f}")
    ''')

    # --- slides 49-50: scaling ------------------------------------------------------------
    nb.md("### Scaling\n\n*Slides: \"Scaling and normalization\" and \"Scaling changes the result "
          "for some model families and not for others\".*")
    explain(
        "Put three differently shaped columns of the slice through three scalers.",
        "Min-max maps onto [0, 1] and one extreme value moves the whole range; standard "
        "scaling subtracts the mean and divides by the standard deviation; robust scaling "
        "uses the median and the interquartile range and survives heavy tails. Distances "
        "and gradients care; trees do not.",
        "Takes speed while moving (skewed), payload (skewed) and inside temperature "
        "(bimodal) from the slice, applies no scaling, min-max, standard (ddof = 0) and "
        "robust scaling, and draws each column's density (a Gaussian kernel estimate on "
        "3,000 readings drawn with the course seed). Unscaled, the three live on axes of "
        "metres per second, kilograms and degrees, so the top row gives each its own panel "
        "and its own unit; the bottom row overlays the three after each scaler, on one "
        "shared, unitless axis. Speed is taken while "
        "moving because over all readings its interquartile range is zero — more than half "
        "the readings are at rest — and robust scaling would divide by zero; the cell "
        "prints that too.",
        "Only the scaled panels are comparable across columns; which scaler is right depends "
        "on the shape, which is why the distribution is measured first — including whether "
        "the scaler is defined at all.")
    nb.figure("scaling", r'''
    all_speed = bus["speed"].astype(float)
    print(f"  speed over all readings: quartiles {all_speed.quantile(0.25):.3f} and "
          f"{all_speed.quantile(0.75):.3f} m/s, so its IQR is {all_speed.quantile(0.75) - all_speed.quantile(0.25):.3f}")
    frame = pd.DataFrame({"speed while moving": all_speed.where(all_speed != 0),
                          "payload": bus["payload"].astype(float),
                          "inside_temperature": bus["inside_temperature"].astype(float)})
    columns = {"speed while moving": BLUE, "payload": ORANGE, "inside_temperature": GREEN}
    units = {"speed while moving": "m/s", "payload": "kg", "inside_temperature": "°C"}
    scalers = {
        "min-max": lambda x: (x - x.min()) / (x.max() - x.min()),
        "standard (ddof = 0)": lambda x: (x - x.mean()) / x.std(ddof=0),
        "robust (median, IQR)": lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)),
    }

    def density(values):
        sample = np.random.default_rng(20200122).choice(values, size=min(3000, len(values)), replace=False)
        grid = np.linspace(values.min(), values.max(), 300)
        return grid, stats.gaussian_kde(sample)(grid)

    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.2, horizontal_spacing=0.07,
                        subplot_titles=[f"no scaling: {c} ({units[c]})" for c in columns] + list(scalers))
    for col, (column, colour) in enumerate(columns.items(), start=1):
        grid, curve = density(frame[column].dropna().to_numpy())
        fig.add_scatter(x=grid, y=curve, mode="lines", line=dict(color=colour, width=2), name=column,
                        row=1, col=col)
        fig.update_xaxes(title_text=units[column], row=1, col=col)
    for col, (label, scale) in enumerate(scalers.items(), start=1):
        for column, colour in columns.items():
            grid, curve = density(scale(frame[column].dropna()).to_numpy())
            fig.add_scatter(x=grid, y=curve, mode="lines", line=dict(color=colour, width=2),
                            showlegend=False, row=2, col=col)
        fig.update_xaxes(title_text="scaled value (no unit)", row=2, col=col)
    fig.update_annotations(font_size=12)
    fig.update_layout(title="Three columns of the slice: each in its own unit (top), and after three scalers (bottom)",
                      legend=dict(orientation="h", y=-0.15))
    show(fig, "scaling", width=1100, height=660)
    for column in columns:
        values = frame[column].dropna()
        print(f"  {column:<20} range {values.min():8.2f} to {values.max():8.2f} {units[column]}")
    ''', slides=["49"], treatment="lab data: the slide's synthetic normal, exponential and uniform "
                                 "columns replaced by the slice's speed while moving, payload and inside "
                                 "temperature")

    # --- slide 51: what a summary hides ---------------------------------------------------
    nb.md("### Window features, and what the first two hide\n\n*Slide: \"Feature extraction in "
          "timeseries\" (mean, standard deviation, minimum, maximum, slope, higher moments).*")
    explain(
        "Find windows of the slice with the same mean and standard deviation and different "
        "movement.",
        "The slide's animation is the \"Datasaurus\": point clouds with identical summary "
        "statistics and different shapes. The same happens in a speed series: two numbers "
        "cannot tell accelerating from braking.",
        "Cuts the slice's speed (time order) into windows of 60 readings (30 seconds, no "
        "gap inside), starting every 10 readings, and keeps those whose mean lies within "
        "0.05 m/s of 2.1 m/s and whose standard deviation lies within 0.05 m/s of 0.9 m/s — "
        "a pair many windows of the slice share. Among them it picks four: the "
        "steepest rise first, then each time the window whose curve is furthest from every "
        "window already chosen (the nearest one's Euclidean distance, maximised), never two "
        "within two minutes of each other — so that no two drawn windows overlap or repeat "
        "one stretch of driving. Draws them and prints all six window features.",
        "Mean and spread agree to within 0.05 m/s, and the four curves are a pull-away from "
        "rest, a burst of speed that fades, a slowdown and recovery, and a burst that ends "
        "at a stop — whose fitted slope, −0.001, is as flat as a steady cruise. Two numbers "
        "cannot tell these apart, and the slope alone misses the last one; that is why the "
        "course's windows carry six features, not two.")
    nb.figure("same_summary_windows", r'''
    speed_in_time = in_time["speed"].to_numpy(float)
    width = 60
    starts = np.arange(0, len(speed_in_time) - width, 10)
    unbroken = np.array([(in_time["_t"].iloc[s + width - 1] - in_time["_t"].iloc[s]).total_seconds() < 31
                         for s in starts])
    windows = np.array([speed_in_time[s:s + width] for s in starts])
    means, sds = windows.mean(axis=1), windows.std(axis=1)
    slopes = np.array([np.polyfit(np.arange(width), w, 1)[0] for w in windows])
    close = np.where(unbroken & (np.abs(means - 2.1) < 0.05) & (np.abs(sds - 0.9) < 0.05))[0]
    chosen = [close[np.argmax(slopes[close])]]                  # the steepest rise first
    while len(chosen) < 4:                                      # then the most different curve each time
        apart = [k for k in close if all(abs(starts[k] - starts[c]) >= 240 for c in chosen)]
        chosen.append(max(apart, key=lambda k: min(np.linalg.norm(windows[k] - windows[c]) for c in chosen)))
    fig = go.Figure()
    for colour, k in zip((RED, ORANGE, GREEN, BLUE), chosen):
        fig.add_scatter(y=windows[k], mode="lines", line=dict(color=colour, width=2),
                        name=f"from {in_time['_t'].iloc[starts[k]]:%d %b %H:%M:%S}: mean {means[k]:.2f}, "
                             f"sd {sds[k]:.2f}, slope {slopes[k]:+.3f}")
    fig.update_layout(title="Four 30-second windows with the same mean and spread, moving differently",
                      xaxis_title="reading in the window (0.5 s apart)", yaxis_title="speed, m/s",
                      legend=dict(orientation="h", y=-0.2))
    show(fig, "same_summary_windows", height=520)
    summary = pd.DataFrame([{"start (UTC)": f"{in_time['_t'].iloc[starts[k]]:%d %b %H:%M:%S}",
                             "mean": means[k], "sd": sds[k], "min": windows[k].min(),
                             "max": windows[k].max(), "slope": slopes[k],
                             "skewness": stats.skew(windows[k])} for k in chosen]).round(3)
    print(summary.to_string(index=False))
    print(f"  windows sharing that mean and spread: {len(close)}; smallest distance between two drawn "
          f"curves: {min(np.linalg.norm(windows[a] - windows[b]) for a in chosen for b in chosen if a < b):.1f} m/s")
    ''', slides=["51"], treatment="lab data: the slide's animated point clouds replaced by four "
                                 "windows of the slice's speed with matching mean and spread")

    # --- slides 52-53: the fitted transform ------------------------------------------------
    nb.md("### The transform is part of the model\n\n*Slides: \"Preprocessing constants are "
          "parameters and must be fitted on training rows only\" and \"Definition — the fitted "
          "transform, and applying it\".*")
    explain(
        "Write the fitted transform and its application.",
        "A median that fills a gap and a mean and standard deviation that scale a column are "
        "learned constants; they are fitted on the training rows, stored, and applied "
        "unchanged to everything afterwards [@kuhn2019, ch. 8]. Recomputing them on the "
        "test rows is leakage [@huyen2022, ch. 4, \"Data Leakage\"].",
        "Writes θ and apply(X, θ) in sympy, with σ the population standard deviation "
        "(ddof = 0) and 1 for a column that does not vary.",
        "Nothing in apply() is measured from the frame in hand.")
    nb.equation("fitted_transform", r'''
    x_ij, med, mu_j, sigma_j = sp.symbols(r"x_{ij} \mathrm{median}_j \mu_j \sigma_j")
    n, i = sp.symbols("n i", positive=True, integer=True)
    theta = sp.Symbol(r"\theta")
    formula(sp.Eq(theta, sp.Function(r"\mathrm{fit}")(sp.Symbol(r"X_{train}")), evaluate=False),
            sp.Eq(theta, sp.Tuple(med, mu_j, sigma_j, sp.Symbol(r"\text{column order}")), evaluate=False))
    formula(sp.Eq(sigma_j, sp.sqrt(sp.Sum((sp.Indexed("x", i) - mu_j)**2, (i, 1, n)) / n), evaluate=False),
            sp.Eq(sp.Symbol(r"\mathrm{apply}(X, \theta)_{ij}"),
                  (sp.Function(r"\mathrm{fill}")(x_ij, med) - mu_j) / sigma_j, evaluate=False))
    ''', slides=["53"])
    explain(
        "Define Lab 3's functions as the solution writes them.",
        "They are the four deliverables of Laboratory 3 — the fit, the application, the leak "
        "detector and the verdict — with the purity statistic both of the last two share "
        "[@kuhn2019, ch. 8; @kaufman2012; @kapoor2023].",
        "Copies the constants (the three verdict thresholds are in the solution, and on the "
        "definition slide, but deliberately not in the stub), the helpers and the four "
        "functions from `solutions/lab_03.py`.",
        "The next cells check the fit against the formula, then use the detector.")
    nb.source(f"{S}/lab_03.py", "LAB", "TARGET", "AGREEMENT", "MAX_LEAK_VALUES", "DDOF",
              "MEMORISED_GAP", "MASK_MISSING_SHARE", "PURITY_FLOOR", "CALLS", "_numeric_columns",
              "fit_preprocessing", "apply_preprocessing", "column_purity", "find_leaks", "_said",
              "keep_or_drop", cite="[@kuhn2019, ch. 8; @kaufman2012; @kapoor2023]")
    explain(
        "Check that the code applies the slide's formula with the stored constants and "
        "nothing else.",
        "The check moves the test set by 1,000 and requires the output to move with it; the "
        "formula says exactly how far.",
        "Fits on the first 70 per cent of the generated day (time order), applies to the "
        "rest, and compares every value with (fill(x, median) − μ)/σ computed from the "
        "stored numbers. Then moves the test rows by 1,000 and checks that the scaled speed "
        "moves by exactly 1,000/σ.",
        "The code is the formula; the moved test set proves no constant was recomputed.")
    nb.code(r'''
    train_rows, test_rows = ordered.iloc[:cut], ordered.iloc[cut:]
    fitted = fit_preprocessing(train_rows)
    applied = apply_preprocessing(test_rows, fitted)
    by_formula = pd.DataFrame({c: (test_rows[c].astype(float).fillna(fitted["medians"][c]) - fitted["means"][c])
                                  / fitted["stds"][c] for c in fitted["columns"]})
    print("largest difference between code and formula:", float((applied - by_formula).abs().max().max()))
    moved = test_rows.copy()
    moved[fitted["columns"]] = moved[fitted["columns"]] + 1000
    shift = float((apply_preprocessing(moved, fitted)["speed"] - applied["speed"]).mean())
    print(f"scaled speed moved by {shift:.3f}; 1000 / σ(speed) = {1000 / fitted['stds']['speed']:.3f}")
    print("training rows' speed σ with ddof = 0:", round(fitted["stds"]["speed"], 4),
          "  pandas' default ddof = 1 would give:", round(float(train_rows["speed"].std()), 4))
    ''')

    # --- slides 54-56: leakage ------------------------------------------------------------
    nb.md("### Target leakage, and the column that is the target\n\n*Slides: \"Definition — target "
          "leakage, and the rule that catches it here\", \"Leakage is detected by scoring each "
          "feature and by asking when it becomes known\" and \"BusID is not correlated with the "
          "target, it is the target under another name\".*")
    explain(
        "Write the leak rule, and prove why it needs its ceiling.",
        "A column leaks when it agrees with the target almost perfectly by presence, or by "
        "value over few distinct values. Purity is one by construction for a column whose "
        "values are all distinct, so without the ceiling every identifier leaks "
        "[@kaufman2012; @kapoor2023].",
        "Writes the rule, agreement and purity in sympy, with the threshold and the ceiling "
        "taken from the solution's own constants (`AGREEMENT`, `MAX_LEAK_VALUES`); then "
        "computes `column_purity` for a row counter (every value distinct) against a random "
        "target, and for the same target against a coin with two values, and evaluates the "
        "rule on both.",
        "The row counter scores 1.0 against a target it knows nothing about; the ceiling of "
        "ten distinct values is what stops the rule from reporting it.")
    nb.equation("leak_rule", r'''
    agreement_c, purity_c, values_c = sp.symbols(r"\mathrm{agreement}(c) \mathrm{purity}(c) |\mathrm{values}(c)|",
                                                 nonnegative=True)
    rule = sp.Or(sp.Ge(agreement_c, AGREEMENT),
                 sp.And(sp.Ge(purity_c, AGREEMENT), sp.Le(values_c, MAX_LEAK_VALUES)))
    same = sp.Symbol(r"\overline{1[c\ \mathrm{present}] = 1[Y{=}\mathrm{IN}]}")
    differ = sp.Symbol(r"\overline{1[c\ \mathrm{present}] \ne 1[Y{=}\mathrm{IN}]}")
    v, n_rows, V = sp.Symbol("v", integer=True), sp.Symbol("n", positive=True), sp.Symbol("V", integer=True)
    formula(sp.Equivalent(sp.Symbol(r"c\ \text{leaks}"), rule, evaluate=False))
    formula(sp.Eq(agreement_c, sp.Max(same, differ), evaluate=False),
            sp.Eq(purity_c, sp.Sum(sp.Function(r"\max_{y}\, n")(v), (v, 1, V)) / n_rows, evaluate=False))
    rng = np.random.default_rng(20200122)
    random_target = pd.Series(rng.random(1000) < 0.5)
    counter_purity = column_purity(pd.Series(np.arange(1000)), random_target)
    coin_purity = column_purity(pd.Series(rng.integers(0, 2, 1000)), random_target)
    print("purity of a row counter against a random target:", counter_purity,
          "— the rule says it leaks:", rule.subs({agreement_c: 0, purity_c: counter_purity, values_c: 1000}))
    print("purity of a random coin against the same target:", round(coin_purity, 3),
          "— the rule says it leaks:", rule.subs({agreement_c: 0, purity_c: coin_purity, values_c: 2}))
    agrees("the leak threshold", AGREEMENT, 0.99, 2)
    agrees("the ceiling on distinct values", MAX_LEAK_VALUES, 10, 0)
    ''', slides=["54"])
    explain(
        "Cross-tabulate `bus_id` against the target, with the deck's figure function.",
        "In the archive `BusID` is filled exactly when the passenger is aboard: 7,723 rows "
        "aboard all carry it and 6,000 not aboard all lack it. The generator reproduces the "
        "leak as `bus_id`.",
        "Copies `figure_leak()` from `make_figs.py` verbatim and feeds it the generated "
        "day's four counts; runs `find_leaks` on the day; prints the archive's counts beside.",
        "Two zeros off the diagonal in both worlds. `find_leaks` names `bus_id`, `label` "
        "(the hand-written label the target was derived from) and `stationary` (the target "
        "inverted), and neither the timestamps nor the phone speed.")
    nb.source(FIGS, "figure_leak")
    explain(
        'Draw the cross-tabulation for the generated day and ask the detector.',
        "The slide's counts are the archive's; the generated day reproduces the leak, with its own counts.",
        "Counts the four cells, calls figure_leak() with them, runs find_leaks() on the day, and prints the archive's counts and absent share beside.",
        'Two zeros off the diagonal in both worlds.')
    nb.figure("leak", r'''
    present, inside = phones["bus_id"].notna(), phones["label2"] == "IN"
    generated_table = {"present_and_aboard": int((present & inside).sum()),
                       "present_and_not_aboard": int((present & ~inside).sum()),
                       "absent_and_aboard": int((~present & inside).sum()),
                       "absent_and_not_aboard": int((~present & ~inside).sum())}
    figure_leak({"busid_against_label": {"value": generated_table}})
    print("  find_leaks on the generated day:", find_leaks(phones, TARGET))
    beside("bus_id against the target", generated_table, f"{archive('busid_against_label')} (archive)",
           "the generated day has 10,800 rows; the archive's labelled day 13,723")
    beside("bus_id absent, per cent", round(100 * float(phones["bus_id"].isna().mean()), 2),
           f"{archive('busid_absent_share')} (archive)",
           "the archive's share is over both days, and its second day is unlabelled and carries "
           "no BusID; the generated first day is 56.3 per cent aboard")
    ''', slides=["56"], treatment="lab data: the slide's function fed the generated day's counts; "
                                 "archive counts printed beside")

    # --- slide 57: the verdict --------------------------------------------------------------
    explain(
        "Check that the verdict's five thresholds are the definition slide's.",
        "Finding a suspect is not deciding what to do with it. The verdict asks, in order: "
        "knowable at prediction time? pure to 99 per cent over ten values or fewer? a quarter "
        "better in sample than out? absent on a fifth of the rows? purity 60 per cent or less "
        "with nothing to mask? — and the first question that answers decides.",
        "Writes the verdict as a sympy Piecewise over the evidence E, in the order the "
        "solution asks its questions and with the solution's constants; evaluates it on "
        "forty-eight planted evidences (every combination of the quantities on either side "
        "of their thresholds) and compares each call with `keep_or_drop`'s; then compares "
        "the constants with the slide's words.",
        "The slide, the solution and the check grade one rule, and the formula on the page "
        "is that rule, case for case.")
    nb.equation("verdict", r'''
    knowable, purity_e, values_e, gap_e, missing_e = sp.symbols(
        r"\mathrm{knowable} \mathrm{purity} |\mathrm{values}| \mathrm{gap} \mathrm{missing}")
    drop, keep = sp.Symbol(r"\text{drop}"), sp.Symbol(r"\text{keep}")
    keep_masked = sp.Symbol(r"\text{keep with the mask}")
    verdict_rule = sp.Piecewise(
        (drop, sp.Not(knowable)),
        (drop, sp.And(sp.Ge(purity_e, AGREEMENT), sp.Le(values_e, MAX_LEAK_VALUES))),
        (drop, sp.Ge(gap_e, MEMORISED_GAP)),
        (keep_masked, sp.Ge(missing_e, MASK_MISSING_SHARE)),
        (drop, sp.Le(purity_e, PURITY_FLOOR)),
        (keep, True), evaluate=False)
    formula(sp.Eq(sp.Function(r"\mathrm{verdict}")(sp.Symbol("E")), verdict_rule, evaluate=False))
    names = {drop: "drop", keep: "keep", keep_masked: "keep with the mask"}
    import itertools
    disagreements, cases = 0, 0
    for k_, p_, n_, g_, m_ in itertools.product((True, False), (0.995, 0.7, 0.5), (2, 1000),
                                                (0.3, 0.0), (0.5, 0.0)):
        evidence = {"missing_share": m_, "imputation_bias_db": 0.0, "purity": p_, "cardinality": n_,
                    "knowable_at_decision_time": k_, "in_sample_minus_out_of_sample": g_}
        by_formula = names[verdict_rule.subs({knowable: k_, purity_e: p_, values_e: n_, gap_e: g_, missing_e: m_})]
        cases += 1
        disagreements += by_formula != keep_or_drop(evidence)[0]
    print(f"planted evidences: {cases}; calls where the formula and keep_or_drop differ: {disagreements}")
    for what, value, stated, places in (("purity threshold", AGREEMENT, 0.99, 2),
                                        ("ceiling on distinct values", MAX_LEAK_VALUES, 10, 0),
                                        ("in-sample advantage that condemns (a quarter)", MEMORISED_GAP, 0.25, 2),
                                        ("absent share that keeps the mask (a fifth)", MASK_MISSING_SHARE, 0.20, 2),
                                        ("purity floor (sixty per cent)", PURITY_FLOOR, 0.60, 2)):
        agrees(what, value, stated, places)
    print("calls:", CALLS)
    ''', slides=["57"])

    # --- Lab 3 -------------------------------------------------------------------------
    nb.md("""
    ## Laboratory 3 — Fit, and find the leak

    *Slide: "Lab 3 — Fit, and find the leak".* Four functions, twenty-five minutes.
    The check moves the test set a thousand units, confirms that the stored constants
    were used, confirms that you found the leak and left the innocent columns alone,
    and grades the verdict on sixteen candidates.
    """)
    nb.statement(f"{LABS}/03_fit_and_leak.py")
    nb.md("""
    **The stub's slide titles, against the deck.** The stub places the lab under "The leak
    in this archive"; no slide has that title. The slide that shows the leak is "BusID is
    not correlated with the target, it is the target under another name". The three
    definition slides it names carry exactly those titles.
    """)
    nb.md("### The solution\n\nThe functions were defined above, where the deck introduces the "
          "transform and the leak rule.")
    explain(
        "Register Lab 3 under the name the Lab 4 solution imports it by.",
        "Lab 4's hand-off does `from lab_03 import fit_preprocessing`.",
        "Puts a module named `lab_03` holding the four functions into `sys.modules`.",
        "The import in Lab 4 will find the functions above.")
    nb.code(r'''
    as_module("lab_03", fit_preprocessing=fit_preprocessing, apply_preprocessing=apply_preprocessing,
              find_leaks=find_leaks, keep_or_drop=keep_or_drop, column_purity=column_purity)
    ''')
    explain(
        "Run the stub's own demonstration against the solved functions.",
        "It is what a student sees when the file is complete.",
        "The stub's `__main__` block, verbatim.",
        "The stored constants, the applied shape, the leaks, and one verdict.")
    nb.step(f"{LABS}/03_fit_and_leak.py", 0)
    nb.md("""
    ### The solution's demonstration, step by step

    What `python3 solutions/lab_03.py` prints, one paragraph of its `__main__` block at a
    time, each verbatim, with two cells of the notebook's own between them: one counts
    what the split by row does at the cut instant, the other recomputes the evidence
    numbers the solution types in.
    """)
    lab3 = f"{S}/lab_03.py"
    explain(
        "Import the two scikit-learn pieces the demonstration uses.",
        "The lab's functions need only pandas and numpy; the demonstration shows the leak's "
        "symptom with a model, so it imports one here and nowhere else.",
        "Imports `LogisticRegression` and `accuracy_score`.",
        "Nothing the check grades depends on scikit-learn.")
    nb.paragraph(lab3, 1)
    explain(
        "Start the demonstration's narrator.",
        "One logger carries every line the solution prints, in the terminal and here.",
        "Creates `say` for Lab 3 and prints the lab's one-line summary.",
        "The seconds at the start of each line are the only output that differs between runs.")
    nb.paragraph(lab3, 2)
    explain(
        "Split the first generated day by time and fit the transform on the training rows.",
        "The first deliverable: medians, means and standard deviations are learned "
        "constants, estimated on the training rows only [@kuhn2019, ch. 8].",
        "Sorts the day by `timestamp_utc`, cuts at `int(len(phones) * 0.7)`, fits "
        "`fit_preprocessing` on the rows before the cut, and shows the stored constants.",
        "Every constant in the table was computed from the training rows; the next cells "
        "apply them to the rest unchanged.")
    nb.paragraph(lab3, 3)
    explain(
        "Count the rows of the cut instant on each side of the split.",
        "Twelve phones share every timestamp in the generated day, so a cut by row number "
        "lands on a boundary between instants only if the row number is a multiple of "
        "twelve. `int(10,800 × 0.7)` should be 7,560 = 630 × 12; floating point makes "
        "0.7 × 10,800 slightly less than 7,560, and `int()` truncates it to 7,559.",
        "Prints the product and the cut, then counts the rows stamped with the first test "
        "instant in the training part and in the test part.",
        "Eleven readings of 09:10:24.597 train and one tests: the split is not "
        "train = {t < t_c}, test = {t ≥ t_c}. Here it costs little — the check and the fit "
        "barely notice one instant — but a model scored on a reading whose eleven "
        "simultaneous neighbours it was trained on is scored on a near-copy, which is the "
        "leak the split exists to prevent. The lab is not changed; Lab 4's `assemble` "
        "moves its cut to the start of a window for exactly this reason. The notebook's own "
        "target-encoding and transform checks earlier in this block copy the lab's cut, so "
        "that their numbers are the lab's; they carry the same straddled instant.")
    nb.code(r'''
    first_test_instant = phones["timestamp_utc"].iloc[cut]
    print(f"0.7 × {len(phones):,} = {0.7 * len(phones)!r}; int() gives cut = {cut:,}")
    print(f"rows stamped {first_test_instant:%H:%M:%S.%f}: in the training rows "
          f"{int((train['timestamp_utc'] == first_test_instant).sum())}, in the test rows "
          f"{int((test['timestamp_utc'] == first_test_instant).sum())}")
    ''')
    explain(
        "Apply the stored constants to the test rows, and move the test rows by 1,000.",
        "The proof that `apply_preprocessing` measures nothing from the frame in hand is "
        "that its output moves exactly with a moved input.",
        "Applies the transform to the test rows; adds 1,000 to every stored column and "
        "applies it again; prints how far the scaled speed moved.",
        "It moves by 1,000/σ(speed) = 1,000/0.9666 = 1,034.5 — had the constants been "
        "recomputed on the moved frame, it would not have moved at all.")
    nb.paragraph(lab3, 4)
    explain(
        "Ask the leak detector, and cross-tabulate `bus_id` against the target.",
        "The third deliverable: `find_leaks` must name the columns that agree with the "
        "target by presence or by value [@kaufman2012].",
        "Runs `find_leaks(phones, TARGET)` and shows the two-by-two table of `bus_id` "
        "present or absent against aboard or not.",
        "`bus_id`, `label` and `stationary` are named; the table has two zeros off the "
        "diagonal.")
    nb.paragraph(lab3, 5)
    explain(
        "Define the scoring function for the symptom.",
        "A leak shows itself as a score too good to be true; a small model makes the "
        "symptom visible.",
        "`score(columns)` fits a logistic regression on the training rows of the chosen "
        "columns (absent values set to 0; `bus_id` as present or not) and returns its "
        "accuracy on the test rows.",
        "The next paragraph calls it with and without `bus_id`.")
    nb.paragraph(lab3, 6)
    explain(
        "Score the model with the leak and without it.",
        "The contrast is the symptom a reviewer sees first.",
        "Scores speed, `rssi1` and `rssi2` alone, and the same three with `bus_id`.",
        "With `bus_id` the model is perfect on the later 30 per cent (1.000); without it, "
        "0.710 — honest, and far from perfect. The perfect score is the symptom, not a "
        "result.")
    nb.paragraph(lab3, 7)
    explain(
        "Draw the cross-tabulation.",
        "The lab's own version of the slide's figure.",
        "A heat map of the four counts; `save_figure` shows it here.",
        "Two empty cells: `bus_id` is the target under another name.")
    nb.paragraph(lab3, 8)
    explain(
        "Prepare the training rows the verdict's evidence is measured on.",
        "A quantity that decides what enters the model must not be computed over the rows "
        "the model is judged on — the rule the medians followed above.",
        "Cuts at the same 70 per cent, keeps the training rows, their aboard indicator, and "
        "a row counter (`reading_id`) to serve as the identifier candidate.",
        "The next two paragraphs build the evidence from these rows.")
    nb.paragraph(lab3, 9)
    explain(
        "Define how one candidate's evidence is assembled.",
        "The verdict takes a dictionary of six quantities; building it in one place keeps "
        "the purity and the cardinality measured, not typed.",
        "`evidence_for` measures the purity against the training rows' target and the "
        "number of distinct values, and passes through the absent share, the fill bias, "
        "whether the column is knowable at prediction time, and the in-sample advantage.",
        "Four of the six quantities are arguments: the next paragraph says where each "
        "comes from.")
    nb.paragraph(lab3, 10)
    explain(
        "Give the verdict on five candidates.",
        "The fourth deliverable: `keep_or_drop` must reach three different calls on "
        "columns that are all pure, or all but pure, and say why in numbers.",
        "Builds the evidence for `bus_id`, `stationary`, `reading_id`, `phone_speed` and "
        "`rssi1`, calls `keep_or_drop` on each, and shows the calls and the reasons.",
        "`bus_id`, `stationary` and the row counter are dropped, for three different "
        "reasons; `phone_speed` is above the purity threshold and kept, because its purity "
        "is spread over 1,415 values; `rssi1` is kept with its mask. Note what is typed "
        "rather than measured in this paragraph: `phone_speed`'s whole evidence (purity "
        "0.9933, 1,415 values, gap 0.0575) and the gaps of `reading_id` (0.5069) and "
        "`rssi1` (0.1214) are numbers, not calls. The next cell recomputes them.")
    nb.paragraph(lab3, 11)
    explain(
        "Recompute the evidence numbers the solution types in.",
        "\"Compute, don't assert\": a verdict built on typed numbers is only as good as "
        "their provenance. The solution does not say how they were measured, so the "
        "notebook states a recipe and checks that it reproduces them.",
        "`phone_speed` is Lab 4's window mean of the phone's speed. Aligns the day (Lab 1), "
        "keeps the hand-off's 1,500 training windows (before the window the 70 per cent "
        "row falls in), and measures purity and distinct values against \"most of the "
        "window's readings aboard\" (and, for comparison, \"any reading aboard\"). The gap "
        "is taken as an unpruned decision tree's accuracy on its training rows minus its "
        "accuracy on the test rows (scikit-learn, seed 20200122): on `reading_id` against "
        "the later row numbers, on `rssi1` with its gaps left empty (the tree handles "
        "them), and on `phone_speed` across the hand-off's split. Every recomputed number "
        "is printed beside the typed one.",
        "All five typed numbers are reproduced to four decimals by this recipe, so the "
        "verdicts rest on measurements — the recipe is the notebook's reconstruction, not "
        "something the solution states. The window target matters: with \"any reading "
        "aboard\" the purity is 0.9940, and the call is the same.")
    nb.code(r'''
    from sklearn.tree import DecisionTreeClassifier


    def memorisation_gap(train_x, train_y, test_x, test_y):
        """An unpruned tree's accuracy on its own rows minus its accuracy on unseen rows."""
        tree = DecisionTreeClassifier(random_state=20200122).fit(train_x, train_y)
        return round(float(tree.score(train_x, train_y) - tree.score(test_x, test_y)), 4)


    labels = (phones[TARGET] == "IN").astype(int).to_numpy()
    windows_table = align(bus, load_phones(), 5)[0].sort_values(["window", "phone_id"]).reset_index(drop=True)
    window_cut = windows_table["window"].iloc[int(len(windows_table) * 0.7)]
    train_windows = windows_table[windows_table["window"] < window_cut]
    test_windows = windows_table[windows_table["window"] >= window_cut]
    majority = lambda frame: (frame["aboard"] >= 0.5).astype(int)
    recomputed = {
        ("phone_speed", "purity"): (0.9933, round(column_purity(train_windows["phone_speed"],
                                                               majority(train_windows).astype(bool)), 4)),
        ("phone_speed", "cardinality"): (1415, int(train_windows["phone_speed"].nunique())),
        ("phone_speed", "gap"): (0.0575, memorisation_gap(train_windows[["phone_speed"]].to_numpy(), majority(train_windows),
                                                          test_windows[["phone_speed"]].to_numpy(), majority(test_windows))),
        ("reading_id", "gap"): (0.5069, memorisation_gap(np.arange(edge).reshape(-1, 1), labels[:edge],
                                                         np.arange(edge, len(phones)).reshape(-1, 1), labels[edge:])),
        ("rssi1", "gap"): (0.1214, memorisation_gap(phones[["rssi1"]].to_numpy()[:edge], labels[:edge],
                                                    phones[["rssi1"]].to_numpy()[edge:], labels[edge:])),
    }
    check = pd.DataFrame([{"candidate": c, "quantity": q, "typed in the solution": typed, "recomputed": mine,
                           "agree": abs(float(typed) - float(mine)) < 5e-5} for (c, q), (typed, mine) in recomputed.items()])
    print(f"training windows: {len(train_windows):,}; test windows: {len(test_windows):,}")
    print(check.to_string(index=False))
    print("phone_speed purity against \"any reading aboard\":",
          round(column_purity(train_windows["phone_speed"], train_windows["aboard"] > 0), 4))
    assert check["agree"].all(), "a typed evidence number is not reproduced"
    ''')
    explain(
        "Print what the check grades.",
        "The demonstration ends by telling the student what `verify/check_03.py` will ask.",
        "One narrated line.",
        "The constants, the moved test set, the leaks with their negative controls, and "
        "the verdict on eight candidates and on eight one-quantity changes.")
    nb.paragraph(lab3, 12)
