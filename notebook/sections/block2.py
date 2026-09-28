"""Block 2 — missingness is a mechanism — and Laboratory 2: the three mechanisms,
the beacons' absence, the mask and the masked moving average, what each fill
invents and what it flattens, the displacement budget (the deck's appendix), and
the Lab 2 solution."""

EX = "Module 2/exercises"
S = f"{EX}/solutions"
LABS = f"{EX}/labs"
FIGS = "Module 2/slides/make_figs.py"


def build(nb, explain) -> None:
    nb.md("""
    ---
    # Block 2 — Missingness is a mechanism

    **The question of this block:** why is a reading absent — and once you fill it,
    what did you invent, and what did you flatten?

    *Slide: "Silver is derived and rebuildable, and bronze must never be modified".*
    The original data (bronze) never change; every cleaned table (silver) is derived,
    disposable and rebuildable, and every decision taken on the way is recorded. The
    working copy this notebook runs in is that rule applied to the notebook itself.
    """)

    # --- slide 32: the three mechanisms --------------------------------------------------
    nb.md("### The three missing-data mechanisms\n\n"
          "*Slide: \"Definition — the three missing-data mechanisms\".*")
    explain(
        "Write the three mechanisms as conditional probabilities.",
        "Whether a gap can be filled honestly depends on what the absence depends on: "
        "nothing (MCAR), values you can see (MAR), or the missing value itself (MNAR) "
        "[@rubin1976, Definitions 1 and 2; @little2019; @vanbuuren2018, § 1.2].",
        "Writes the three statements as sympy relations, with M the absence indicator, "
        "Y_obs what is seen and Y_mis what is not: MCAR and MAR are equalities, and MNAR is "
        "the case where the dependence on Y_mis does not drop out.",
        "Signal strength falls with distance and a beacon is heard only when it is near, so "
        "a reading is absent because it was weak: the third case.")
    nb.equation("mechanisms", r'''
    P = sp.Function("P")
    given_both = P(sp.Symbol(r"M \mid Y_{obs}, Y_{mis}"))
    given_seen = P(sp.Symbol(r"M \mid Y_{obs}"))
    formula(sp.Symbol(r"\text{MCAR:}"), sp.Eq(given_both, P(sp.Symbol("M")), evaluate=False))
    formula(sp.Symbol(r"\text{MAR:}"), sp.Eq(given_both, given_seen, evaluate=False))
    formula(sp.Symbol(r"\text{MNAR:}"), sp.Ne(given_both, given_seen, evaluate=False))
    ''', slides=["32"])
    explain(
        "Apply each mechanism to the same true signal and see what a mean fill does.",
        "The slide's picture is a generated poster of the three cases. The generator holds "
        "the true signal strength of every reading, heard or not, so the three mechanisms "
        "can be imposed on one column and their consequences measured "
        "[@vanbuuren2018, § 1.3.3].",
        "Takes beacon 1's true signal on the first generated day and three masks of about "
        "the real one's size (88.7 per cent absent). MCAR: a random mask. MAR needs an "
        "observed column that carries the signal, and the generated phones have none — "
        "their speed and their other beacons are drawn independently of beacon 1's "
        "distance — so the cell makes one, labelled illustrative: the distance to the "
        "beacon as a satellite fix would report it, the true distance plus 5 metres of "
        "normal error (a stated choice; the lab data carry no fix error). The MAR mask "
        "drops a reading with a probability that depends on that observed distance only, "
        "rising with its rank from about 0.79 for the nearest reading to about 0.99 for the "
        "farthest, so that every tenth of the distance keeps some readings. MNAR: the "
        "generator's own mask, where the weak readings are the absent ones. For each mask "
        "the cell computes two fills' bias on the absent rows: the mean of the kept "
        "readings, and the mean of the kept readings *in the same tenth of the observed "
        "distance* — conditioning on what is seen [@vanbuuren2018, § 1.2].",
        "Under MCAR both fills are unbiased. Under MAR the mean fill is too strong, "
        "because the kept readings are the near ones, and conditioning on the observed "
        "distance repairs it: within a tenth of the distance the kept and the absent "
        "readings are alike. Under MNAR the mean fill is more than ten decibels too strong, "
        "and conditioning shrinks the error without removing it, because within a tenth of "
        "the distance the absent readings are still the weaker ones. The mechanism, not "
        "the method, decides — and the archive is the third case.")
    nb.figure("three_mechanisms", r'''
    signal_true = truth["rssi1_true"].to_numpy(float)
    real_mask = phones["rssi1"].isna().to_numpy()
    rate = real_mask.mean()
    rng = np.random.default_rng(20200122)
    # Illustrative: an observed distance, as a satellite fix with 5 m of error would give it.
    seen_distance = np.clip(truth["rssi1_distance_true"].to_numpy(float) + rng.normal(0, 5.0, len(truth)), 0.5, None)
    far_rank = stats.rankdata(seen_distance) / len(seen_distance)            # 0 near … 1 far
    p_absent = rate - 0.1 + 0.2 * far_rank                                   # 0.79 … 0.99
    masks = {"MCAR (a random mask)": rng.random(len(truth)) < rate,
             "MAR (absent where the observed distance is far)": rng.random(len(truth)) < p_absent,
             "MNAR (weak = absent)": real_mask}
    tenth = np.minimum((far_rank * 10).astype(int), 9)                       # tenths of the observed distance

    def fill_biases(mask):
        mean_fill = float(signal_true[~mask].mean() - signal_true[mask].mean())
        kept_by_tenth = pd.Series(signal_true[~mask]).groupby(tenth[~mask]).mean()
        conditional = pd.Series(tenth[mask]).map(kept_by_tenth).to_numpy()
        return mean_fill, float(np.mean(conditional - signal_true[mask])), int(pd.Series(~mask).groupby(tenth).sum().min())

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(masks), shared_yaxes=True)
    rows_out = {}
    for col, (label, mask) in enumerate(masks.items(), start=1):
        for rows, name, colour in ((~mask, "kept (heard)", BLUE), (mask, "absent: the hidden truth", ORANGE)):
            fig.add_histogram(x=signal_true[rows], histnorm="probability density", nbinsx=40,
                              marker_color=colour, opacity=0.6, name=name, showlegend=col == 1,
                              row=1, col=col)
        mean_fill, conditional, fewest = fill_biases(mask)
        fig.update_xaxes(title_text=f"true dBm — bias: mean fill {mean_fill:+.1f} dB,<br>"
                                    f"fill within a tenth of the distance {conditional:+.1f} dB", row=1, col=col)
        rows_out[label] = {"absent share": round(float(mask.mean()), 3), "mean fill, dB": round(mean_fill, 2),
                           "conditional fill, dB": round(conditional, 2), "fewest kept rows in a tenth": fewest}
    fig.update_annotations(font_size=12)
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=-0.3), margin=dict(l=60, r=30, t=80, b=130),
                      title="One true signal, three mechanisms: conditioning on what is seen repairs MAR, not MNAR")
    show(fig, "three_mechanisms", height=500)
    print(pd.DataFrame(rows_out).T.to_string())
    print(f"  correlation of the observed distance's logarithm with the true signal: "
          f"{np.corrcoef(np.log10(seen_distance), signal_true)[0, 1]:.3f}")
    beside("absence of the vehicle beacon (rssi1), archive", f"{archive('beacon_absent_share')['rssi1']} %",
           "\"the vehicle beacon is silent on 57% to 86% of rows\" (the slide's poster)",
           "the range 57 to 86.3 per cent spans all five beacons; the vehicle beacon alone is "
           "silent on 57.0 per cent")
    ''', slides=["32"], treatment="lab data: the slide's generated poster replaced by the three "
                                 "mechanisms imposed on the generator's true signal; the MAR "
                                 "driver (an observed distance with 5 m of error) is illustrative")

    # --- slide 33: two causes of absence ---------------------------------------------
    explain(
        "Draw the geometry behind an absent reading.",
        "The slide's diagram places a phone, a vehicle and a beacon on a map: the phone's "
        "distance to the vehicle by satellite positioning (d_GPS) and its distance to a "
        "beacon (d_BLE). A reading exists only if the phone is in range, the packet "
        "arrives, and the signal registers; out of range is informative, a lost packet is "
        "not [@servizi2023].",
        "Draws the loop from the slice; stop A, the stop inferred on the route map where "
        "the doors opened most often; the vehicle's reported position at the first reading "
        "on the loop's north edge on 23 January; and a phone placed 10 metres from the stop "
        "(an illustrative position — the lab data have no phone positions), with both "
        "distances. The slide's diagram draws the phone between two shuttles. The lab "
        "slice holds one shuttle, so a second vehicle position cannot be drawn from data; "
        "the second end of the picture is a stop instead, which is the comparison that "
        "matters in Block 2 — aboard against waiting nearby.",
        "Every distance on this map is tens of metres at most: the scale at which proximity "
        "has to separate aboard from waiting.")
    nb.figure("phone_bus_geometry", r'''
    east, north = to_metres(bus["lat"], bus["lon"])
    positions = np.column_stack(to_metres(in_time["lat"], in_time["lon"]))
    on_north_edge = np.where((positions[:, 1] > 60) & (in_time["_t"].dt.date.astype(str) == "2020-01-23"))[0]
    vehicle = positions[on_north_edge[0]]
    stop = stops.loc["stop A", ["east", "north"]].to_numpy(float)
    phone = stop + np.array([8.0, 6.0])                     # illustrative: 10 m from the stop
    d_gps, d_ble = np.linalg.norm(phone - vehicle), np.linalg.norm(phone - stop)
    fig = go.Figure()
    fig.add_scatter(x=east[::20], y=north[::20], mode="markers", marker=dict(color=GREY, size=2),
                    name="the loop (slice positions)")
    moment = in_time["_t"].iloc[on_north_edge[0]]
    for point, label, colour, symbol in ((vehicle, f"vehicle, {moment:%d %b %H:%M:%S} UTC", BLUE, "square"),
                                         (stop, "stop A (inferred: the doors opened here most often)", GREEN, "diamond"),
                                         (phone, "phone (illustrative position)", ORANGE, "circle")):
        fig.add_scatter(x=[point[0]], y=[point[1]], mode="markers", name=label,
                        marker=dict(color=colour, size=14, symbol=symbol))
    for end, label in ((vehicle, f"d_GPS = {d_gps:.0f} m"), (stop, f"d_BLE = {d_ble:.0f} m")):
        fig.add_scatter(x=[phone[0], end[0]], y=[phone[1], end[1]], mode="lines+text",
                        line=dict(color=NAVY, dash="dash"), text=["", label],
                        textposition="top center", showlegend=False)
    fig.update_layout(title="Two distances from one phone: to the vehicle's fix, and to a beacon",
                      xaxis_title="metres east", yaxis_title="metres north", yaxis_scaleanchor="x",
                      legend=dict(orientation="h", y=-0.15))
    show(fig, "phone_bus_geometry", width=800, height=700)
    print(f"  d_GPS = {d_gps:.1f} m, d_BLE = {d_ble:.1f} m")
    ''', slides=["33"], treatment="lab data with an illustrative phone: the slide's diagram redrawn "
                                 "on the slice's loop, vehicle position and inferred stop; the "
                                 "slide's second shuttle is not in the slice")

    # --- slide 34: absence per beacon ---------------------------------------------------
    nb.md("### How much is absent, and how it is written down\n\n"
          "*Slides: \"Most beacons carry no reading on most rows, and absence is encoded twice\" "
          "and \"The vehicle beacon is heard more often when the passenger is not aboard\".*")
    explain(
        "Draw each beacon's share of absent readings with the deck's own function, fed the "
        "lab data.",
        "The slide measures the archive: 57 to 86.3 per cent of rows carry no reading. The "
        "generator was calibrated to be heard at the archive's rate *within each aboard "
        "state*, which reproduces the association with the label rather than the marginal "
        "shares, so the lab data's shares differ.",
        "Copies `figure_absence()` from `make_figs.py` verbatim and calls it with the "
        "generated day's shares in place of the archive's; prints the archive's beside. "
        "Then checks, on the generated rows, that the empty signal strength and the "
        "proximity of −1 mark the same rows.",
        "Most readings are absent in both worlds; the exact shares are the archive's, and "
        "are labelled so.")
    nb.source(FIGS, "figure_absence")
    explain(
        "Draw the absence per beacon on the generated day, and print the archive's shares beside.",
        "The slide's chart is the archive's; the lab data can only show their own.",
        "Calls figure_absence() with the generated shares in the shape measured.json uses, then prints the archive's recorded shares and checks the double encoding on the generated rows.",
        'Absent more often than present on every beacon, in both worlds.')
    nb.figure("beacon_absence", r'''
    generated_absence = {b: round(float(phones[b].isna().mean()) * 100, 1)
                         for b in ["rssiA", "rssiB", "rssiC", "rssi1", "rssi2"]}
    figure_absence({"beacon_absent_share": {"value": generated_absence}})
    beside("absent per cent, per beacon", generated_absence, f"{archive('beacon_absent_share')} (archive)",
           "the generator matches the archive's heard rates per aboard state, not its marginal shares")
    beside("range across the five beacons, per cent",
           f"{min(generated_absence.values())} to {max(generated_absence.values())}",
           "57 to 86.3 (archive)", "as above")
    encoded_twice = all((phones[r].isna() == (phones[p] == -1)).all() for r, p in PROXIMITY.items())
    print(f"  absence encoded twice, and the two agree on every generated row: {encoded_twice} "
          f"(archive: {archive('absence_encoded_twice')})")
    ''', slides=["34"], treatment="lab data: the slide's function fed the generated day's shares; "
                                 "archive shares printed beside")
    explain(
        "Test the natural inference — an unheard vehicle beacon means not aboard — on the "
        "labelled rows.",
        "The slide measured it on the archive and refuted it: the vehicle beacon is heard on "
        "8.8 per cent of aboard rows and 14.4 per cent of not-aboard rows, and \"heard\" "
        "agrees with \"aboard\" less often than the base rate of 56.3 per cent. The "
        "generator was calibrated to those rates, so the lab data can repeat the test.",
        "For each beacon on the first generated day: the share heard when aboard, when not "
        "aboard, and the agreement of \"heard\" with \"aboard\"; against the base rate.",
        "Answering \"aboard\" every time beats every beacon. A real mechanism does not make "
        "a useful feature; measure before relying on the physics.")
    nb.code(r'''
    labelled = phones[phones["label2"].notna()]
    aboard = labelled["label2"] == "IN"
    rows = {}
    for b in ["rssiA", "rssiB", "rssiC", "rssi1", "rssi2"]:
        heard = labelled[b].notna()
        rows[b] = {"heard when aboard %": round(100 * heard[aboard].mean(), 1),
                   "heard when not aboard %": round(100 * heard[~aboard].mean(), 1),
                   "agreement with aboard %": round(100 * (heard == aboard).mean(), 1)}
    table = pd.DataFrame(rows).T
    print(table.to_string())
    base_rate = 100 * aboard.mean()
    agrees("base rate: per cent of labelled rows aboard", base_rate, archive("aboard_share_of_labelled"), 1)
    agrees("vehicle beacon heard when aboard, per cent", table.loc["rssi1", "heard when aboard %"], 8.8, 1)
    agrees("vehicle beacon heard when not aboard, per cent", table.loc["rssi1", "heard when not aboard %"], 14.4, 1)
    print(f"  every beacon's agreement is below the base rate: "
          f"{bool((table['agreement with aboard %'] < base_rate).all())}")
    beside("agreement range across beacons, per cent",
           f"{table['agreement with aboard %'].min()} to {table['agreement with aboard %'].max()}",
           "42.4 to 47.5 (archive)", "generated rows reproduce the archive's rates to within a "
           "tenth of a point, not exactly")
    beside("labelled rows", f"{len(labelled):,}", f"{archive('labelled_rows'):,} (archive)",
           "the generator's first day has 12 phones x 900 readings")
    ''')

    # --- slide 36: the mechanism, measured ------------------------------------------------
    nb.md("### The absent readings are the weak ones\n\n"
          "*Slide: \"The absent readings are the weak ones, which are the distant ones\".*")
    explain(
        "Draw the true signal of the heard and the absent readings.",
        "The archive never recorded the absent readings; the generator knows them. The two "
        "distributions barely overlap: this is what MNAR looks like [@little2019].",
        "Copies `figure_mechanism()` from `make_figs.py` verbatim and calls it with the "
        "truth-kept first day.",
        "Any fill made from the visible readings averages the near ones only.")
    nb.source(FIGS, "figure_mechanism")
    explain(
        'Draw the true signal of the heard and the absent readings.',
        "The slide's figure is generated data, so it can be redrawn exactly.",
        'Calls figure_mechanism() with the truth-kept first day and prints both means.',
        'The absent readings are 13.6 decibels weaker on average — which is exactly the bias of filling them with the mean of the heard ones.')
    nb.figure("mechanism", r'''
    figure_mechanism(truth)
    heard = truth["rssi1"].notna()
    print(f"  true signal, heard readings: mean {truth.loc[heard, 'rssi1_true'].mean():.1f} dBm; "
          f"absent readings: mean {truth.loc[~heard, 'rssi1_true'].mean():.1f} dBm")
    ''', slides=["36"], treatment="exact: the slide's own code")

    # --- slides 37-39: bias, mask, EMA ---------------------------------------------------
    nb.md("### Imputation bias, the mask, and the masked moving average\n\n"
          "*Slides: \"Definition — imputation bias\", \"A fill must be recorded, because a filled "
          "value and a measured value look alike\" and \"Definition — the absence mask and the "
          "masked exponential moving average\".*")
    explain(
        "Write the bias of a fill.",
        "The bias is the mean amount by which the invented values sit above the truth, over "
        "the rows the method actually filled; positive means too strong — the phone placed "
        "nearer the beacon than it was [@little2019; @vanbuuren2018].",
        "Writes the definition in sympy, with F the set of filled rows.",
        "\"Drop\" fills nothing, so it scores zero by construction, not by merit.")
    nb.equation("imputation_bias", r'''
    j, size_F = sp.Symbol("j", integer=True), sp.Symbol("|F|", positive=True, integer=True)
    t = sp.IndexedBase("t")
    fill, truth_t = sp.Function(r"\mathrm{fill}"), sp.Function(r"\mathrm{truth}")
    bias = sp.Sum(fill(t[j]) - truth_t(t[j]), (j, 1, size_F)) / size_F
    row = sp.Symbol("t")
    filled_rows = sp.ConditionSet(row, sp.And(sp.Eq(sp.Function("m")(row), 1),
                                              sp.Symbol(r"\text{the method filled } t")), sp.Naturals)
    formula(sp.Eq(sp.Symbol(r"\mathrm{bias}"), bias, evaluate=False), sp.Symbol(r"\text{(dB)}"))
    formula(sp.Eq(sp.Symbol("F"), filled_rows, evaluate=False),
            sp.Symbol(r"\text{its rows numbered } j = 1, \dots, |F| \text{ in the sum}"))
    ''', slides=["37"])
    explain(
        "Write the mask and the masked moving average.",
        "The mask is the only record of which values are invented; the archive writes the "
        "same absence twice (an empty signal and a proximity of −1), so the mask must "
        "recognise both [@vanbuuren2018, § 1.3.7]. The masked average smooths only the "
        "heard readings and carries its last value across a gap. The study behind the "
        "archive filled BLE gaps with an exponentially weighted moving average and told the "
        "classifier where it could not [@servizi2023, Appendix B], and the scalable "
        "detector kept imputation and masking [@servizi2026, Table 1]. One difference must "
        "be said: Appendix B describes its average as taken over a time window \"where "
        "points close to the center window have a higher weight\" — a centred window, which "
        "looks at readings after the gap as well as before — while the slide, the lab and "
        "the solution use the causal recursion below, which looks only backwards. The lab "
        "files attribute that recursion to Servizi et al. (2023) in their docstrings, which "
        "are quoted verbatim here and not changed; the attribution is to the method's "
        "family (an exponentially weighted average with a mask), not to this exact form.",
        "Writes m_t as a sympy Piecewise and the recursion s_t with α = 0.3, and unrolls "
        "the recursion three steps to show the weights it gives to past readings.",
        "Three choices are printed beside the recursion — α = 0.3 stated not tuned, the "
        "plain recursion, one series per phone — and each is visible in the code below. "
        "The unrolled weights decay backwards in time only: a causal filter, usable at "
        "prediction time, unlike a centred window.")
    nb.equation("mask_and_ema", r'''
    alpha = sp.Symbol(r"\alpha", positive=True)
    s, step = sp.Function("s"), sp.Symbol("t", integer=True)
    x_t = sp.IndexedBase("x")
    rssi_null, prox_t = sp.Symbol(r"\mathrm{rssi}_t\ \text{is empty}"), sp.Symbol(r"\mathrm{prox}_t")
    absence_mask = sp.Piecewise((1, sp.Or(rssi_null, sp.Eq(prox_t, -1))), (0, True))
    recursion = sp.Eq(s(step), alpha * x_t[step] + (1 - alpha) * s(step - 1), evaluate=False)
    formula(sp.Eq(sp.Symbol("m_t"), absence_mask, evaluate=False), recursion,
            sp.Symbol(r"\text{over the heard readings only}"), sp.Eq(alpha, sp.Rational(3, 10), evaluate=False))
    unrolled = recursion.rhs
    for back in (1, 2):
        unrolled = unrolled.subs(s(step - back), alpha * x_t[step - back] + (1 - alpha) * s(step - back - 1))
    weighted = sum(alpha * (1 - alpha)**back * x_t[step - back] for back in range(3)) + (1 - alpha)**3 * s(step - 3)
    formula(sp.Eq(s(step), weighted, evaluate=False))
    print("the recursion unrolled three times equals the weighted sum:", sp.expand(unrolled - weighted) == 0)
    print("weights on x_t, x_{t-1}, x_{t-2} at α = 0.3:",
          [sp.expand(unrolled).coeff(x_t[step - back]).subs(alpha, sp.Rational(3, 10)) for back in range(3)])
    ''', slides=["39"])
    explain(
        "Define Lab 2's functions as the solution writes them.",
        "They are the three deliverables of Laboratory 2 and the functions the deck's "
        "imputation figures are computed with; the deck imports them rather than "
        "reimplementing them.",
        "Copies the constants, the two helpers `_missing` and `_ema_masked`, and "
        "`impute_with_mask`, `imputation_bias` and `fills_are_biased_which_way` from "
        "`solutions/lab_02.py`.",
        "The next cells check the recursion against the slide's formula and then draw the "
        "slides' figures with these functions.")
    nb.source(f"{S}/lab_02.py", "LAB", "BEACONS", "PROXIMITY", "SENTINEL", "SMOOTHING", "_missing",
              "_ema_masked", "impute_with_mask", "imputation_bias", "fills_are_biased_which_way",
              cite="[@rubin1976; @little2019; @vanbuuren2018; @servizi2023]")
    explain(
        "Check that the solution's masked average is the slide's recursion.",
        "The check grades the recursion on a planted two-phone gap; the same test can be "
        "written symbolically.",
        "Plants two phones: A hears −60, misses two readings, hears −70, misses one; B hears "
        "−80, misses one, hears −75. Absence is written both ways (empty signal, proximity "
        "−1). The slide's recursion, evaluated with sympy at α = 0.3, predicts each filled "
        "value; `impute_with_mask(…, \"ema_masked\")` must return them.",
        "A's last gap is filled with 0.3·(−70) + 0.7·(−60) = −63: the formula and the code "
        "agree to the last digit, and B's gap is filled from B alone.")
    nb.code(r'''
    planted = pd.DataFrame({"phone_id": ["A", "B", "A", "A", "B", "A", "B", "A"],
                            "value": [-60, -80, None, None, None, -70, -75, None]})
    for b in BEACONS:
        planted[b] = planted["value"]
        planted[PROXIMITY[b]] = np.where(planted["value"].isna(), SENTINEL, 2)
    filled = impute_with_mask(planted, "ema_masked")

    a = sp.Rational(3, 10)
    expected = []
    state = {}
    for phone, value in zip(planted["phone_id"], planted["value"]):
        if pd.notna(value):
            state[phone] = sp.Integer(int(value)) if phone not in state else a * int(value) + (1 - a) * state[phone]
            expected.append(float(value))                  # heard rows keep their own value
        else:
            expected.append(float(state[phone]))           # a gap takes the last state
    planted["filled by the code"] = filled["rssi1_filled"]
    planted["the slide's recursion"] = expected
    planted["mask"] = filled["rssi1_missing"]
    print(planted[["phone_id", "value", "mask", "filled by the code", "the slide's recursion"]].to_string())
    assert np.allclose(planted["filled by the code"], planted["the slide's recursion"])
    print("formula and code agree on every row")
    ''')
    explain(
        "Register Lab 2 under the name the deck's figure code imports it by.",
        "`make_figs.py` computes the imputation figures with `from lab_02 import "
        "impute_with_mask, imputation_bias`.",
        "Puts a module named `lab_02` holding the functions above into `sys.modules`.",
        "The deck's code, run below, uses the functions on this page.")
    nb.code(r'''
    as_module("lab_02", impute_with_mask=impute_with_mask, imputation_bias=imputation_bias,
              fills_are_biased_which_way=fills_are_biased_which_way)
    ''')

    # --- slides 40-42: what a fill invents, and what it flattens ------------------------
    nb.md("### What each fill invents\n\n*Slides: \"Every imputation places the absent readings "
          "nearer the beacon than they were\" and \"Definition — what a fill (imputation) does to "
          "the spread\".*")
    explain(
        "Compute the three fills with the functions the lab is graded on, the deck's way.",
        "The slide's numbers — 13.6 decibels for the mean, 9 for the masked average — are "
        "computed by `make_figs.py`'s `imputation()`, which calls Lab 2's functions.",
        "Copies `imputation()` and `figure_imputation_by_distance()` verbatim.",
        "The next cell calls them.")
    nb.source(FIGS, "imputation", "figure_imputation_by_distance")
    explain(
        "Draw the slide's figure: one volunteer against the true distance, and the bias.",
        "The left panel shows where each fill puts the phone against where it really was; "
        "the right panel is each method's bias over the whole day.",
        "Calls `imputation(truth, phones)` and `figure_imputation_by_distance(...)`.",
        "Mean-filling invents 13.6 decibels on 88.7 per cent of the rows; the masked "
        "average 9.0 — better, and still wrong, because it carries a near reading forward.")
    nb.figure("imputation_by_distance", r'''
    filled_by_method, biases = imputation(truth, phones)
    figure_imputation_by_distance(truth, phones, filled_by_method, biases)
    agrees("mean fill, bias in decibels", biases["mean"], 13.6, 1)
    agrees("masked moving average, bias in decibels", biases["ema_masked"], 9.0, 1)
    agrees("drop, bias in decibels", biases["drop"], 0.0, 1)
    agrees("share of beacon 1's rows filled, per cent",
           100 * filled_by_method["mean"]["rssi1_missing"].mean(), 88.7, 1)
    print("  the fills come out:", fills_are_biased_which_way())
    ''', slides=["40", "41"], treatment="exact: the slide's own code")
    explain(
        "Derive what a single constant fill leaves of the spread.",
        "The slide states that filling a share f of the rows with one constant leaves √(1 − f) "
        "of the standard deviation, when the kept readings are a fair sample. Little and "
        "Rubin give the sample-variance version, a factor (n_obs − 1)/(n − 1) "
        "[@little2019, § 4.2.1]; multiple imputation restores the lost variation "
        "[@rubin1987].",
        "Writes the population variance of a column whose n·f filled rows sit at the "
        "kept mean m: the kept rows contribute n(1 − f)·v, the filled rows nothing. sympy "
        "simplifies the ratio of standard deviations, then evaluates it at f = 0.887.",
        "√(1 − f) = 0.34 at 88.7 per cent filled: the dotted line on the next figure.")
    nb.equation("spread_kept", r'''
    f, n, v = sp.symbols("f n v", positive=True)
    variance_filled = (n * (1 - f) * v + n * f * 0) / n          # the filled rows sit on the mean
    ratio = sp.sqrt(sp.simplify(variance_filled / v))
    sd = sp.Function(r"\mathrm{sd}")
    formula(sp.Eq(sp.Symbol(r"\text{spread kept}"), sd(sp.Symbol(r"\text{filled}")) / sd(sp.Symbol(r"\text{truth}")),
                  evaluate=False),
            sp.Eq(sp.Symbol(r"\text{constant fill}"), ratio, evaluate=False))
    share_filled = float(filled_by_method["mean"]["rssi1_missing"].mean())
    agrees("√(1 − f) at the filled share", float(ratio.subs(f, share_filled)), 0.34, 2)
    ''', slides=["41"])
    explain(
        "Draw bias and spread side by side, with the deck's code.",
        "Bias is half the story: a fill can be unbiased and still flatten the column. The "
        "slide reports what each fill kept of the standard deviation.",
        "Copies `spreads_after_filling()` and `figure_spread()` from `make_figs.py` "
        "verbatim, and calls them.",
        "Mean-filling keeps 0.36 of the spread, the masked average 0.87; dropping *raises* "
        "it to 1.07, because the rows that survive are not a random sample. The gap between "
        "0.36 and the 0.34 a fair sample would give is the mechanism, seen from the spread.")
    nb.source(FIGS, "spreads_after_filling", "figure_spread")
    explain(
        "Draw the slide's bias-and-spread figure and check its numbers.",
        'The slide quotes 0.36, 0.87 and 1.07 of the spread kept, the 0.34 a fair sample would give, and 88.7 per cent filled.',
        'Calls spreads_after_filling() and figure_spread(), compares every number with the slide, and squares the ratios to show the variance kept.',
        'All five agree; the word "variance" on the slide is the one slip, recorded below.')
    nb.figure("imputation_spread", r'''
    ratios, filled_share, predicted = spreads_after_filling(truth, filled_by_method)
    figure_spread(ratios, biases, predicted)
    agrees("spread kept, mean fill", ratios["mean"], 0.36, 2)
    agrees("spread kept, masked moving average", ratios["ema_masked"], 0.87, 2)
    agrees("spread kept, drop", ratios["drop"], 1.07, 2)
    agrees("spread kept if the absence were completely at random", predicted, 0.34, 2)
    agrees("filled share, per cent", filled_share, 88.7, 1)
    beside("masked moving average: share of the variance kept", f"{ratios['ema_masked'] ** 2:.2f}",
           "\"preserves 87% of the original variance\"",
           "0.87 is a ratio of standard deviations; the variance kept is its square. The same "
           "slide's \"36% of the original variation\" and the misconceptions poster's \"destroys "
           "64% of variation\" are standard deviations too (variance kept: "
           f"{ratios['mean'] ** 2:.2f})")
    ''', slides=["42"], treatment="exact: the slide's own code")

    # --- slide 43 and the appendix: the displacement budget ------------------------------
    nb.md("""
    ### The readings that do arrive, and the displacement budget

    *Slides: "The readings that do arrive carry an error that can be written down" and
    the deck's appendix, "Appendix — The displacement budget, term by term".*

    Absence is one fault; the readings that arrive are also wrong by an amount that can
    be accounted for. Signal strength is disturbed by a body between phone and beacon,
    by which pocket holds the phone, by the angles and by the mounting height. A
    position is wrong by the satellite fix's error plus the distance moved since the
    last fix.
    """)
    explain(
        "Write the budget, and compare it with the quadrature sum the appendix mentions.",
        "The appendix states Δd = ΔGPS + ΔSpeed·ΔT: the fix's error plus how far the phone "
        "could have moved unobserved, added as magnitudes — a worst case. Independent "
        "errors combine in quadrature instead. The appendix says the budget is the author's "
        "own derivation, with no source beyond the arithmetic.",
        "Writes both totals in sympy and proves the linear sum is never smaller (the "
        "difference of squares is 2·ΔGPS·ΔSpeed·ΔT ≥ 0). Then evaluates the movement term "
        "with lab numbers: the shuttle's top speed on the slice times the phone's reporting "
        "interval.",
        "Between two phone readings the shuttle can move about 3.5 metres; the satellite "
        "error is not in the lab data and stays a symbol.")
    nb.equation("displacement_budget", r'''
    d_gps, d_speed, d_t = sp.symbols(r"\Delta_{GPS} \Delta_{Speed} \Delta_T", nonnegative=True)
    worst = d_gps + d_speed * d_t
    quadrature = sp.sqrt(d_gps**2 + (d_speed * d_t)**2)
    formula(sp.Eq(sp.Symbol(r"\Delta d"), worst, evaluate=False),
            sp.Eq(sp.Symbol(r"\Delta d_{\perp}"), quadrature, evaluate=False))
    print("worst² − quadrature² =", sp.expand(worst**2 - quadrature**2), "  (never negative)")
    top_speed = float(bus["speed"].abs().max())
    interval = float(archive("phone_interval_s"))
    print(f"movement term with the slice's top speed {top_speed:.3f} m/s and the phone interval "
          f"{interval} s: {top_speed * interval:.2f} m")
    ''', slides=["95"])

    # --- Lab 2 -------------------------------------------------------------------------
    nb.md("""
    ## Laboratory 2 — Missingness is a mechanism

    *Slide: "Lab 2 — Missingness is a mechanism".* Three functions, twenty-five
    minutes. The check holds the true values you cannot see: your mask must recognise
    both encodings of absence, your bias must reproduce its arithmetic, and your
    direction must follow from the mechanism.
    """)
    nb.statement(f"{LABS}/02_the_mechanism.py")
    nb.md("""
    **The stub's slide titles, against the deck.** The stub places the lab under "The
    mask, and why it must survive"; no slide has that title. The slide that makes the
    point is "A fill must be recorded, because a filled value and a measured value look
    alike". The three definition slides it names carry exactly those titles. The stub
    attributes the masked moving average to Servizi et al. (2023); as noted where the
    recursion is written above, that paper's Appendix B describes a centre-weighted
    window average, and the lab's causal recursion is a member of the same family rather
    than the paper's exact method.
    """)
    nb.md("### The solution\n\nThe five functions were defined above, where the deck introduces "
          "the mask and the recursion. First, the lab file's own demonstration.")
    explain(
        "Run the stub's own demonstration against the solved functions.",
        "It is what a student sees when the file is complete.",
        "The stub's `__main__` block, verbatim.",
        "For beacon 1: the masked rows and the bias of each method, and the direction.")
    nb.step(f"{LABS}/02_the_mechanism.py", 0)
    nb.md("""
    ### The solution's demonstration, step by step

    What `python3 solutions/lab_02.py` prints, one paragraph of its `__main__` block at a
    time, each verbatim. It measures what each fill invented against the hidden truth and
    draws the lab's version of the slide's figure.
    """)
    lab2 = f"{S}/lab_02.py"
    explain(
        "Start the demonstration's narrator.",
        "One logger carries every line the solution prints, in the terminal and here.",
        "Creates `say` for Lab 2 and prints the lab's one-line summary.",
        "The seconds at the start of each line are the only output that differs between runs.")
    nb.paragraph(lab2, 1)
    explain(
        "Load the student's view and the truth, and measure the absence.",
        "The bias of a fill can only be measured against values nobody recorded; the "
        "generator kept them in a second frame.",
        "Loads the first generated day twice — as the student sees it, and with its hidden "
        "columns — then builds the mask with `_missing` and prints its share, checking that "
        "the empty signal and the proximity of −1 mark the same rows.",
        "Beacon 1 is absent on 88.7 per cent of the rows, and the two encodings agree on "
        "every one.")
    nb.paragraph(lab2, 2)
    explain(
        "Fill three ways and score each against the truth.",
        "These are the lab's three deliverables in one loop: `impute_with_mask`, "
        "`imputation_bias`, and the direction `fills_are_biased_which_way` names.",
        "For drop, mean and the masked moving average: fills beacon 1, computes the mean of "
        "fill − truth over the filled rows, prints each, tabulates the three, and prints the "
        "named direction.",
        "+0.0, +13.6 and +9.0 decibels: dropping invents nothing by construction; both fills "
        "are too strong, because the readings you can see are the near ones.")
    nb.paragraph(lab2, 3)
    explain(
        "Draw one volunteer against the true distance, and the three biases.",
        "The table says how wrong; the picture says why — the fills are made of near "
        "readings and stand in for far ones.",
        "For the first phone: the heard readings, the hidden truth of the absent ones, the "
        "masked average carried forward, and the mean of the heard readings, all against "
        "the true distance to beacon 1; beside it, the three biases as bars.",
        "Every fill sits above the grey cloud of the truth it replaces, and the further "
        "the phone was, the further above.")
    nb.paragraph(lab2, 4)
    explain(
        "Print what the check grades.",
        "The demonstration ends by telling the student what `verify/check_02.py` will ask.",
        "One narrated line.",
        "The mask is graded on both encodings, the recursion on a planted gap, the bias on "
        "planted triples and on the day, and the direction against the truth.")
    nb.paragraph(lab2, 5)
