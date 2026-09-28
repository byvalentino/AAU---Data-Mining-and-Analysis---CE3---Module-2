"""Block 4 — the series, and what it costs — and Laboratory 4: stationarity,
window features, feature engineering from time and position, selection and
reduction, the Fourier transform, the split by time, the nearest match, the
hand-off table, and the Lab 4 solution."""

EX = "Module 2/exercises"
S = f"{EX}/solutions"
LABS = f"{EX}/labs"
FIGS = "Module 2/slides/make_figs.py"


def build(nb, explain) -> None:
    nb.md("""
    ---
    # Block 4 — The series, and what it costs

    **The question of this block:** what can a window of readings say that one
    reading cannot, what does a split that respects time cost, and what does a join
    cost as its tolerance widens?
    """)

    # --- slides 62-65: stationarity ---------------------------------------------------
    nb.md("### Stationarity\n\n*Slides: \"Stationary vs non-stationary time series\" (three "
          "slides) and \"Most time-series methods assume stationarity without stating the "
          "assumption\".*")
    explain(
        "Redraw the slides' two series and their forecasts: a stationary AR(1) and a series "
        "with a trend and two seasonal cycles.",
        "A stationary series has a mean, a variance and an autocorrelation that do not "
        "depend on when it is observed; its forecast returns to the mean with a band that "
        "stops growing [@box2015]. A trending, seasonal series carries its pattern forward.",
        "Simulates 50 steps of x_t = 0.7·x_{t−1} + ε_t (σ = 0.6), and of a random walk "
        "with drift 0.08 a step (σ = 0.25) plus cycles of 12 and 7 steps (course seed), and "
        "draws each 20-step forecast with its 50, 80 and 95 per cent bands in closed form: "
        "for the AR(1), mean φʰ·x_T and standard deviation σ√((1 − φ²ʰ)/(1 − φ²)); for the "
        "walk, mean x_T + 0.08·h plus the cycles, and standard deviation σ√h. A third panel "
        "applies two of the slide's repairs to the non-stationary series — subtract the "
        "seasonal profile (known here, because the series is simulated), then difference — "
        "and draws what is left. The slides' animation does not state its parameters, so "
        "these are illustrative.",
        "The AR(1) band widens towards the series' own spread and stops; the walk's band "
        "keeps widening with the square root of the horizon, around a forecast that carries "
        "the trend and the cycles on for ever. After the two repairs the series is a "
        "constant (the drift) plus noise — stationary, and modellable. A model fitted on a "
        "drifting mean extrapolates the drift; the logarithm, the slide's third repair, is "
        "for a variance that grows with the level, which this simulation does not have.")
    nb.figure("stationarity_synthetic", r'''
    rng = np.random.default_rng(20200122)
    steps, horizon, phi, noise = 50, 20, 0.7, 0.6
    ar = np.zeros(steps)
    for k in range(1, steps):
        ar[k] = phi * ar[k - 1] + rng.normal(0, noise)
    ahead = np.arange(1, horizon + 1)
    ar_mean = ar[-1] * phi ** ahead
    ar_sd = noise * np.sqrt((1 - phi ** (2 * ahead)) / (1 - phi ** 2))
    clock = np.arange(steps + horizon)
    drift, walk_noise = 0.08, 0.25
    cycles = 0.5 * np.sin(2 * np.pi * clock / 12) + 0.3 * np.sin(2 * np.pi * clock / 7)
    level = np.cumsum(drift + rng.normal(0, walk_noise, steps))           # random walk with drift
    trending = level + cycles[:steps]
    walk_mean = level[-1] + drift * ahead + cycles[steps:]
    walk_sd = walk_noise * np.sqrt(ahead)
    repaired = np.diff(trending - cycles[:steps])                          # deseasonalise, then difference
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.09, subplot_titles=(
        "Stationary: AR(1), φ = 0.7 (illustrative)",
        "Non-stationary: random walk with drift, and two cycles (illustrative)",
        "The same series repaired: seasonal profile subtracted, then differenced"))
    future = np.arange(steps, steps + horizon)
    for row, series, centre, spread, colour in ((1, ar, ar_mean, ar_sd, GREEN),
                                                (2, trending, walk_mean, walk_sd, BLUE)):
        fig.add_scatter(x=np.arange(steps), y=series, mode="lines", line=dict(color=colour),
                        showlegend=False, row=row, col=1)
        for z, alpha in ((1.96, 0.15), (1.2816, 0.25), (0.6745, 0.35)):
            fig.add_scatter(x=np.r_[future, future[::-1]], y=np.r_[centre + z * spread, (centre - z * spread)[::-1]],
                            fill="toself", mode="none", fillcolor=f"rgba(80,80,80,{alpha})",
                            showlegend=False, row=row, col=1)
        fig.add_scatter(x=future, y=centre, mode="lines", line=dict(color=colour, width=3),
                        showlegend=False, row=row, col=1)
    fig.add_scatter(x=np.arange(1, steps), y=repaired, mode="lines", line=dict(color=ORANGE),
                    showlegend=False, row=3, col=1)
    fig.add_hline(y=float(repaired.mean()), line=dict(color=GREY, dash="dash"), row=3, col=1,
                  annotation_text=f"mean {repaired.mean():.2f} (the drift is {drift})")
    fig.update_xaxes(title_text="time step", range=[0, steps + horizon], row=3, col=1)
    fig.update_layout(title="Forecasts with 50, 80 and 95 per cent bands, and one repair")
    show(fig, "stationarity_synthetic", height=820)
    print(f"  95% band half-width at 1 and 20 steps ahead: AR(1) {1.96 * ar_sd[0]:.2f} and {1.96 * ar_sd[-1]:.2f} "
          f"(limit {1.96 * noise / np.sqrt(1 - phi ** 2):.2f}); walk {1.96 * walk_sd[0]:.2f} and {1.96 * walk_sd[-1]:.2f}")
    half = len(repaired) // 2
    print(f"  repaired series: mean {repaired[:half].mean():.3f} in its first half, {repaired[half:].mean():.3f} "
          f"in its second; standard deviation {repaired[:half].std():.3f} and {repaired[half:].std():.3f}")
    ''', slides=["62", "63", "64"], treatment="illustrative: the slides' animated forecasts "
                                             "redrawn static from a seeded simulation, with "
                                             "the deseasonalise-and-difference repair")
    explain(
        "Look for the same properties in the slice.",
        "The repairs the deck lists — difference a trend away, subtract a seasonal profile, "
        "take a logarithm to steady a growing variance — only make sense after the series is "
        "tested [@box2015].",
        "Averages the slice per minute (gaps removed, so the two days sit end to end). Draws "
        "the outside temperature, its first difference, and the speed's 15-minute rolling "
        "mean and standard deviation on 22 January; prints the temperature of each day's "
        "first and last half hour.",
        "The temperature climbs through each day and resets overnight — a trend with a daily "
        "cycle — while its first difference hovers at zero apart from the jump between the "
        "days. The speed's mean and spread change with the service pattern. Neither column "
        "is stationary as recorded.")
    nb.figure("stationarity_on_slice", r'''
    per_minute = in_time.set_index("_t")[["outside_temperature", "speed"]].resample("1min").mean().dropna()
    first = per_minute[per_minute.index.date.astype(str) == "2020-01-22"]
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.09, subplot_titles=(
        "outside temperature, per-minute mean (both days end to end, by minute of recording)",
        "its first difference",
        "speed on 22 January: 15-minute rolling mean and standard deviation"))
    fig.add_scatter(x=np.arange(len(per_minute)), y=per_minute["outside_temperature"], mode="lines",
                    line=dict(color=ORANGE), showlegend=False, row=1, col=1)
    fig.add_scatter(x=np.arange(len(per_minute) - 1), y=per_minute["outside_temperature"].diff().dropna(),
                    mode="lines", line=dict(color=GREY), showlegend=False, row=2, col=1)
    fig.add_scatter(x=first.index, y=first["speed"].rolling(15).mean(), mode="lines",
                    line=dict(color=BLUE), name="rolling mean", row=3, col=1)
    fig.add_scatter(x=first.index, y=first["speed"].rolling(15).std(ddof=0), mode="lines",
                    line=dict(color=RED), name="rolling standard deviation", row=3, col=1)
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="°C per minute", row=2, col=1)
    fig.update_yaxes(title_text="m/s", row=3, col=1)
    fig.update_layout(title="Stationarity tested on the slice", legend=dict(orientation="h", y=-0.1))
    show(fig, "stationarity_on_slice", height=760)
    for day, part in per_minute.groupby(per_minute.index.date.astype(str)):
        print(f"  {day}: outside temperature {part['outside_temperature'].head(30).mean():.1f} °C in the "
              f"first half hour, {part['outside_temperature'].tail(30).mean():.1f} °C in the last")
    ''', slides=["62", "63", "64"], treatment="lab data: the stationarity properties measured on "
                                             "the slice (temperature trend, speed rolling "
                                             "statistics)")

    # --- slides 66-67: the six window features -------------------------------------
    nb.md("### Six features from a window\n\n*Slides: \"A window of readings yields six cheap "
          "features that one reading cannot\" and \"Definition — six window features\"; the lag-k "
          "autocorrelation's definition slide is hidden, and the stub states it in full.*")
    explain(
        "Write the six features, including the slope and the lag-1 autocorrelation.",
        "Mean and standard deviation give position and spread; minimum, maximum and slope "
        "give shape; the autocorrelation gives structure [@kuhn2019, ch. 5; @nielsen2019]. "
        "The autocorrelation is the Box–Jenkins estimator — one mean, one sum of squares — "
        "which is not what `pandas.Series.autocorr` computes [@box2015, § 2.1.4].",
        "Writes the slope and r_k in sympy.",
        "Both are one line of arithmetic; the next cells check the code against them.")
    nb.equation("window_features", r'''
    i, w, k, t = sp.symbols("i w k t", positive=True, integer=True)
    x = sp.IndexedBase("x")
    ibar, xbar = sp.Symbol(r"\bar{i}"), sp.Symbol(r"\bar{x}")
    slope = sp.Sum((i - ibar) * (x[i] - xbar), (i, 1, w)) / sp.Sum((i - ibar)**2, (i, 1, w))
    r_k = sp.Sum((x[t] - xbar) * (x[t - k] - xbar), (t, k + 1, w)) / sp.Sum((x[t] - xbar)**2, (t, 1, w))
    formula(sp.Eq(xbar, sp.Sum(x[i], (i, 1, w)) / w, evaluate=False),
            sp.Eq(sp.Symbol(r"\sigma"), sp.sqrt(sp.Sum((x[i] - xbar)**2, (i, 1, w)) / w), evaluate=False),
            sp.Symbol(r"\text{(ddof = 0)}"),
            sp.Eq(sp.Symbol(r"x_{\min}"), sp.Function(r"\min_{i}")(x[i]), evaluate=False),
            sp.Eq(sp.Symbol(r"x_{\max}"), sp.Function(r"\max_{i}")(x[i]), evaluate=False))
    formula(sp.Eq(sp.Symbol("b"), slope, evaluate=False), sp.Eq(sp.Symbol("r_k"), r_k, evaluate=False))
    ''', slides=["67"])
    explain(
        "Define the first two of Lab 4's functions as the solution writes them.",
        "`window_features` is the first deliverable of Laboratory 4; it uses exactly the "
        "last `width` values, because a window that reaches further back is a small leak.",
        "Copies the constants, `sample_autocorrelation` and `window_features` from "
        "`solutions/lab_04.py`.",
        "The remaining two functions are defined where the deck introduces the split and "
        "the join.")
    nb.source(f"{S}/lab_04.py", "LAB", "TOLERANCES", "DDOF", "TRAIN_FRACTION", "TARGET_COLUMN",
              "HANDOFF_SCHEMA", "sample_autocorrelation", "window_features",
              cite="[@box2015, § 2.1.4; @kuhn2019, ch. 5]")
    explain(
        "Check the code against the formulas.",
        "The hidden definition slide says the estimator returns exactly two fifths on the "
        "window 1, 2, 3, 4, 5; the slope formula is the least-squares gradient.",
        "Evaluates r_1 on 1…5 with sympy (an exact fraction) and with "
        "`sample_autocorrelation`; evaluates the slope formula with sympy on a planted "
        "window and compares it with `window_features`, which uses `numpy.polyfit`.",
        "Two fifths both ways, and pandas' estimator gives 1.0 on the same window — the "
        "difference the stub warns about.")
    nb.code(r'''
    planted = [1, 2, 3, 4, 5]
    exact = r_k.subs({k: 1, w: 5}).doit().subs({x[m]: v for m, v in enumerate(planted, start=1)})
    exact = sp.nsimplify(exact.subs(xbar, sp.Rational(sum(planted), len(planted))))
    print("r_1 of 1..5 by the formula:", exact, "  by the code:", sample_autocorrelation(planted, 1),
          "  by pandas:", pd.Series(planted).autocorr(1))
    agrees("r_1 of the window 1, 2, 3, 4, 5", sample_autocorrelation(planted, 1), 0.4, 10)
    window = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    values = {x[m]: v for m, v in enumerate(window, start=1)}
    by_formula = slope.subs(w, len(window)).doit().subs(values).subs(
        {ibar: sp.Rational(len(window) + 1, 2), xbar: sp.Rational(int(sum(window)), len(window))})
    print("slope of", window, "by the formula:", float(by_formula), "  by the code:",
          window_features(pd.Series(window), len(window))["slope"])
    ''')
    explain(
        "Draw one window of the slice's speed and its six features.",
        "The slide's picture is a generated drawing of a sine-like signal; the slice has a "
        "real one. It also says the autocorrelation is \"measured here at 0.9967\" — that is "
        "the whole day's lag-1 value, not a window's.",
        "Takes 22 January from 10:15 UTC, marks the window of the last 60 readings "
        "(30 seconds) ending at the 180th reading, computes `window_features`, and draws the "
        "mean, the ±1 standard deviation band, the extremes and the fitted slope. Then "
        "computes r_1 over every 30-second window of the moving day.",
        "Six numbers describe the window instant by instant; the slope turns braking into a "
        "number an operator can act on.")
    nb.figure("window_anatomy", r'''
    stretch = slice_day_one[slice_day_one["_t"] >= pd.Timestamp("2020-01-22 10:15:00", tz="UTC")].head(240)
    series = stretch["speed"].reset_index(drop=True)
    end, width = 180, 60
    feats = window_features(series.iloc[:end], width)
    position = np.arange(end - width, end)
    fig = go.Figure()
    fig.add_scatter(y=series, mode="lines", line=dict(color=GREY, width=1.5), name="speed, 0.5 s apart")
    fig.add_vrect(x0=end - width, x1=end - 1, fillcolor="rgba(42,120,214,0.08)", line_width=0,
                  annotation_text=f"the last w = {width} readings", annotation_position="top left")
    fig.add_scatter(x=position, y=np.full(width, feats["mean"]), mode="lines", line=dict(color=BLUE, width=3),
                    name=f"mean {feats['mean']:.2f} m/s")
    fig.add_scatter(x=np.r_[position, position[::-1]],
                    y=np.r_[np.full(width, feats["mean"] + feats["std"]), np.full(width, feats["mean"] - feats["std"])],
                    fill="toself", mode="none", fillcolor="rgba(42,120,214,0.15)",
                    name=f"± sd {feats['std']:.2f} (ddof = 0)")
    centre = (width - 1) / 2
    fig.add_scatter(x=position, y=feats["mean"] + feats["slope"] * (np.arange(width) - centre), mode="lines",
                    line=dict(color=ORANGE, width=3, dash="dash"), name=f"slope {feats['slope']:+.4f} per reading")
    inside = series.iloc[end - width:end]
    for value, label, colour in ((feats["minimum"], "minimum", RED), (feats["maximum"], "maximum", GREEN)):
        at = int(inside[inside == value].index[0])
        fig.add_scatter(x=[at], y=[value], mode="markers", marker=dict(size=12, color=colour),
                        name=f"{label} {value:.2f}")
    fig.update_layout(title=f"One window of the slice's speed and its features (r₁ = {feats['autocorrelation_1']:.3f})",
                      xaxis_title="reading", yaxis_title="speed, m/s", legend=dict(orientation="h", y=-0.2))
    show(fig, "window_anatomy", height=520)
    print({key: round(value, 4) for key, value in feats.items()})
    moving = slice_day_one[slice_day_one["speed"].abs() > 0]["speed"].to_numpy()
    window_r1 = [sample_autocorrelation(moving[s:s + width], 1) for s in range(0, len(moving) - width, width)]
    beside("lag-1 autocorrelation of a 30-second window", f"median {np.nanmedian(window_r1):.3f} over "
           f"{len(window_r1)} windows of the moving day", "0.9967 (the drawing: \"measured here at 0.9967\")",
           "0.9967 is the whole day's lag-1 value (Module 1); within a short window the mean and the "
           "sum of squares are the window's own, and r₁ is lower")
    ''', slides=["67"], treatment="lab data: the slide's generated drawing replaced by a window of "
                                 "the slice's speed and window_features()")

    # --- slide 68: peaks and amplitude ------------------------------------------------
    explain(
        "Compute the slide's other window features on a real window: peaks, amplitude, "
        "peak-to-peak, and the points beyond one standard deviation.",
        "The slide lists these features over a time interval and illustrates amplitude with "
        "a textbook sine [@nielsen2019].",
        "Takes five minutes of 22 January's speed, finds peaks with `scipy.signal.find_peaks` "
        "(prominence 0.5 m/s, a stated choice), and computes amplitude (half the range), "
        "peak-to-peak, the peaks above mean + 1 sd, and the readings outside ±1 sd.",
        "Each is one line of code and one more column; each must still be justified like any "
        "other feature.")
    nb.figure("peaks_amplitude", r'''
    five = slice_day_one[slice_day_one["_t"] >= pd.Timestamp("2020-01-22 10:15:00", tz="UTC")].head(600)["speed"].to_numpy()
    peaks, _ = signal.find_peaks(five, prominence=0.5)
    mean, sd = five.mean(), five.std()
    fig = go.Figure()
    fig.add_scatter(y=five, mode="lines", line=dict(color=BLUE, width=1.5), name="speed")
    fig.add_scatter(x=peaks, y=five[peaks], mode="markers", marker=dict(color=RED, size=9), name=f"{len(peaks)} peaks")
    for level, label in ((mean + sd, "mean + 1 sd"), (mean, "mean"), (mean - sd, "mean − 1 sd")):
        fig.add_hline(y=level, line=dict(color=GREY, dash="dot"), annotation_text=label)
    fig.update_layout(title="Five minutes of speed: peaks, amplitude and the one-sd band",
                      xaxis_title="reading (0.5 s apart)", yaxis_title="m/s", legend=dict(orientation="h", y=-0.2))
    show(fig, "peaks_amplitude", height=460)
    print(f"  peaks: {len(peaks)}; above mean + 1 sd: {int((five[peaks] > mean + sd).sum())}; "
          f"readings outside ±1 sd: {int((np.abs(five - mean) > sd).sum())} of {len(five)}; "
          f"amplitude {(five.max() - five.min()) / 2:.2f} m/s; peak-to-peak {five.max() - five.min():.2f} m/s")
    ''', slides=["68"], treatment="lab data: the slide's textbook sine replaced by five minutes of "
                                 "the slice's speed")

    # --- slides 69-71: feature engineering basics --------------------------------------
    nb.md("### Feature engineering from columns, from time, and from position\n\n*Slides: "
          "\"Feature engineering basics\" (three slides).*")
    nb.md("The next three figures use one-minute windows of the slice, which the next cell builds.")
    explain(
        "Summarise the slice per minute.",
        "Combining columns, extracting date parts and correlating features all need one row "
        "per unit; a minute is long enough to average and short enough to keep the day's "
        "structure.",
        "Groups the slice by minute (UTC), keeps minutes with at least 100 readings, and "
        "computes `window_features` on the speed plus mean payload, inside and outside "
        "temperature, battery level, the share driven manually, and the mean of the "
        "per-reading product payload × speed.",
        "A table of one row per minute, used by the next five figures.")
    nb.code(r'''
    rows = []
    for minute, part in in_time.groupby(in_time["_t"].dt.floor("1min")):
        if len(part) < 100:
            continue
        feats = window_features(part["speed"], len(part))
        rows.append({"minute": minute, "day": str(minute.date()), **{f"speed_{k}": v for k, v in feats.items()},
                     "payload": part["payload"].mean(), "inside_temperature": part["inside_temperature"].mean(),
                     "outside_temperature": part["outside_temperature"].mean(),
                     "battery_level": part["battery_level"].mean(), "manual_share": (part["mode"] == "manual").mean(),
                     "mean_of_product": (part["payload"] * part["speed"]).mean()})
    minutes = pd.DataFrame(rows)
    print(f"{len(minutes)} one-minute windows; columns: {', '.join(minutes.columns[2:])}")
    ''')
    explain(
        "Combine two columns into one feature, and see what it adds.",
        "The slide's example multiplies two columns (distance to a fire station and a "
        "complexity index). The slice's analogue: payload times speed, how much load is "
        "moving. Domain knowledge chooses the combination; no algorithm will.",
        "Plots the per-minute product of the means against each factor, and against the mean "
        "of the per-reading product.",
        "The product follows speed closely and payload hardly at all (correlations printed), "
        "so it carries a claim the two factors do not state separately. The product of two means "
        "and the mean of the product are different definitions; here they track each other, "
        "but the feature has to say which it is.")
    nb.figure("combined_feature", r'''
    minutes["payload_x_speed"] = minutes["payload"] * minutes["speed_mean"]
    pairs = [("speed_mean", "mean speed, m/s"), ("payload", "mean payload, kg"),
             ("mean_of_product", "mean of payload × speed per reading")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[p[1] for p in pairs], shared_yaxes=True)
    for col, (column, label) in enumerate(pairs, start=1):
        fig.add_scatter(x=minutes[column], y=minutes["payload_x_speed"], mode="markers",
                        marker=dict(color=BLUE, size=5, opacity=0.6), showlegend=False, row=1, col=col)
        fig.update_xaxes(title_text=label, row=1, col=col)
    fig.update_yaxes(title_text="mean payload × mean speed, kg·m/s", row=1, col=1)
    fig.update_layout(title="A combined feature against its two factors, per minute of the slice")
    show(fig, "combined_feature", height=440)
    corr = minutes[["payload_x_speed", "speed_mean", "payload", "mean_of_product"]].corr().round(3)
    print(corr.loc["payload_x_speed"].to_string())
    ''', slides=["69"], treatment="lab data: the slide's insurance-data scatter replaced by payload "
                                 "× speed on the slice's minutes")
    explain(
        "Extract a date part: the hour of the day, in local time.",
        "The slide shows time-of-day, day-of-week, time-of-month and time-of-year bins. The "
        "slice spans two days, so only the hour is informative here, and it must be "
        "extracted in local time — computed from `utc_time`, never read from `timestamp` "
        "[@mckinney2022, ch. 11].",
        "Converts `utc_time` to Europe/Copenhagen, extracts the hour, and counts readings "
        "per hour and day.",
        "Both days ran from the 9 o'clock to the 13 o'clock hour, local time, and the "
        "second day stopped early in the last of them; a model given the hour learns the timetable, which is what the "
        "feature asserts.")
    nb.figure("date_parts", r'''
    local = pd.to_datetime(bus["utc_time"], utc=True).dt.tz_convert("Europe/Copenhagen")
    per_hour = pd.crosstab(local.dt.hour, local.dt.date.astype(str))
    fig = go.Figure()
    for day, colour in zip(per_hour.columns, (BLUE, ORANGE)):
        fig.add_bar(x=per_hour.index, y=per_hour[day], name=day, marker_color=colour)
    fig.update_layout(barmode="group", title="Readings per local hour of the day (Europe/Copenhagen)",
                      xaxis_title="hour of the day, local time", yaxis_title="readings",
                      legend=dict(orientation="h", y=-0.2))
    show(fig, "date_parts", height=420)
    print(per_hour.to_string())
    ''', slides=["70"], treatment="lab data: the slide's date-part diagram replaced by the hour of "
                                 "day extracted from the slice's utc_time")
    explain(
        "Write the four motion channels derived from positions.",
        "The slide shows the channels a GPS-based mode detector stacks — speed, acceleration, "
        "jerk and bearing rate — each a difference quotient of the one before "
        "[@dabiri2018].",
        "Writes them in sympy, with d(P₁, P₂) the distance between consecutive fixes "
        "(the slide uses Vincenty's formula; below the haversine formula on a sphere, "
        "which differs by far less than a metre over these distances).",
        "Four channels from two columns and a clock: position is a rich source of features.")
    nb.equation("gps_channels", r'''
    dt = sp.Symbol(r"\Delta t", positive=True)
    S1, S2, A1, A2 = sp.symbols("S_{P_1} S_{P_2} A_{P_1} A_{P_2}")
    formula(sp.Eq(S1, sp.Function("d")(sp.Symbol("P_1"), sp.Symbol("P_2")) / dt, evaluate=False),
            sp.Eq(A1, (S2 - S1) / dt, evaluate=False),
            sp.Eq(sp.Symbol("J_{P_1}"), (A2 - A1) / dt, evaluate=False),
            sp.Eq(sp.Symbol("BR_{P_1}"), sp.Abs(sp.Symbol(r"\mathrm{Bearing}_{P_2}") - sp.Symbol(r"\mathrm{Bearing}_{P_1}")), evaluate=False))
    ''', slides=["71"])
    explain(
        "Derive the four channels from the slice's positions, and compare the derived speed "
        "with the speed the vehicle reports.",
        "Deriving a channel from positions is only as good as the positions; the vehicle "
        "reports its own speed and heading, so the derivation can be checked.",
        "Takes two minutes of 22 January from 10:15 UTC, every fourth reading (about two "
        "seconds apart), computes haversine distances, speed, acceleration, jerk and the "
        "bearing rate, and draws them with the reported speed.",
        "The derived speed follows the reported magnitude; each further difference is noisier. "
        "The bearing is undefined when the vehicle stands or reverses, so the bearing rate "
        "spikes at every stop — a channel needs a rule for rest before it is a feature.")
    nb.figure("gps_channels", r'''
    track = slice_day_one[slice_day_one["_t"] >= pd.Timestamp("2020-01-22 10:15:00", tz="UTC")].head(240).iloc[::4]
    lat, lon = np.radians(track["lat"].to_numpy()), np.radians(track["lon"].to_numpy())
    seconds = (track["_t"] - track["_t"].iloc[0]).dt.total_seconds().to_numpy()
    R = 6_371_000.0                                             # mean Earth radius, metres
    h = np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
    distance = 2 * R * np.arcsin(np.sqrt(h))
    dt_s = np.diff(seconds)
    speed_ch = distance / dt_s
    accel = np.diff(speed_ch) / dt_s[1:]
    jerk = np.diff(accel) / dt_s[2:]
    bearing = np.degrees(np.arctan2(np.sin(np.diff(lon)) * np.cos(lat[1:]),
                                    np.cos(lat[:-1]) * np.sin(lat[1:]) - np.sin(lat[:-1]) * np.cos(lat[1:]) * np.cos(np.diff(lon))))
    bearing_rate = np.abs((np.diff(bearing) + 180) % 360 - 180)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("speed, m/s", "acceleration, m/s²", "jerk, m/s³", "bearing rate, degrees per fix"))
    fig.add_scatter(x=seconds[1:], y=speed_ch, mode="lines", line=dict(color=BLUE), name="derived from positions", row=1, col=1)
    fig.add_scatter(x=seconds, y=track["speed"].abs(), mode="lines", line=dict(color=GREY, dash="dot"),
                    name="reported by the vehicle", row=1, col=1)
    fig.add_scatter(x=seconds[2:], y=accel, mode="lines", line=dict(color=ORANGE), showlegend=False, row=2, col=1)
    fig.add_scatter(x=seconds[3:], y=jerk, mode="lines", line=dict(color=RED), showlegend=False, row=3, col=1)
    fig.add_scatter(x=seconds[2:], y=bearing_rate, mode="lines", line=dict(color=GREEN), showlegend=False, row=4, col=1)
    fig.update_xaxes(title_text="seconds after 10:15 UTC, 22 January", row=4, col=1)
    fig.update_layout(title="Four channels derived from two minutes of positions",
                      legend=dict(orientation="h", y=-0.08))
    show(fig, "gps_channels", height=760)
    reported = track["speed"].abs().to_numpy()[1:]
    print(f"  derived against reported speed: correlation {np.corrcoef(speed_ch, reported)[0, 1]:.3f}, "
          f"median absolute difference {np.median(np.abs(speed_ch - reported)):.3f} m/s")
    ''', slides=["71"], treatment="lab data: the slide's diagram of the four channels computed on the "
                                 "slice's positions")

    # --- slides 72-73 and 84: selection and reduction --------------------------------
    nb.md("### Selection and reduction\n\n*Slides: \"Feature selection vs dimensionality reduction\" "
          "(two slides) and \"Selection keeps columns and reduction replaces them, and they trade "
          "differently\".*")
    explain(
        "Compare the Pearson and Spearman correlations of the slice's minute features.",
        "A correlation filter needs its own choice: Pearson measures straight-line agreement, "
        "Spearman agreement in rank, which survives a curved relationship.",
        "Computes both matrices over the minute table's ten features and draws them side by "
        "side.",
        "Where the two disagree, the relationship is monotone but not straight — and a filter "
        "built on the wrong one keeps or drops the wrong column.")
    nb.figure("correlation_matrices", r'''
    feature_columns = ["speed_mean", "speed_std", "speed_maximum", "speed_slope", "speed_autocorrelation_1",
                       "payload", "inside_temperature", "outside_temperature", "battery_level", "manual_share"]
    pearson = minutes[feature_columns].corr(method="pearson")
    spearman = minutes[feature_columns].corr(method="spearman")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pearson", "Spearman"), horizontal_spacing=0.18)
    for col, matrix in ((1, pearson), (2, spearman)):
        fig.add_heatmap(z=matrix.to_numpy(), x=feature_columns, y=feature_columns, zmin=-1, zmax=1,
                        colorscale="RdBu_r", text=matrix.round(2).to_numpy(), texttemplate="%{text}",
                        showscale=col == 2, row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Correlations between the slice's minute features, two ways",
                      margin=dict(l=160, r=30, t=70, b=160))
    show(fig, "correlation_matrices", width=1200, height=640)
    gap = (spearman - pearson).abs()
    np.fill_diagonal(gap.values, 0)
    worst = gap.stack().idxmax()
    print(f"  largest Pearson-Spearman disagreement: {worst[0]} vs {worst[1]}: "
          f"Pearson {pearson.loc[worst]:.2f}, Spearman {spearman.loc[worst]:.2f}")
    ''', slides=["72"], treatment="lab data: the slide's insurance-data matrices replaced by the "
                                 "slice's minute features")
    explain(
        "Reduce the minute features with principal component analysis.",
        "Reduction replaces columns with combinations of them: compact, and no longer "
        "explainable to a colleague.",
        "Keeps the minutes where all ten features are defined, standardises them "
        "(ddof = 0), takes the singular value decomposition, "
        "plots every minute on the first two components coloured by day, and draws each "
        "feature's loading as an arrow.",
        "Two components carry about half the variance (the title prints the shares); each is "
        "a mixture of speed, temperature and battery that no one can read off a dashboard.")
    nb.figure("pca", r'''
    complete = minutes.dropna(subset=feature_columns).reset_index(drop=True)
    print(f"  minutes with every feature defined: {len(complete)} of {len(minutes)} "
          "(a minute at rest has no autocorrelation: its sum of squares is zero)")
    standard = (complete[feature_columns] - complete[feature_columns].mean()) / complete[feature_columns].std(ddof=0)
    u, singular, vt = np.linalg.svd(standard.to_numpy(), full_matrices=False)
    explained = singular ** 2 / np.sum(singular ** 2)
    scores = standard.to_numpy() @ vt[:2].T
    fig = go.Figure()
    for day, colour in (("2020-01-22", BLUE), ("2020-01-23", ORANGE)):
        chosen = (complete["day"] == day).to_numpy()
        fig.add_scatter(x=scores[chosen, 0], y=scores[chosen, 1], mode="markers",
                        marker=dict(color=colour, size=5, opacity=0.6), name=day)
    scale = np.abs(scores).max() * 0.8
    for column, (a, b) in zip(feature_columns, vt[:2].T):
        fig.add_scatter(x=[0, a * scale], y=[0, b * scale], mode="lines+text", text=["", column],
                        textposition="top center", line=dict(color=GREY), showlegend=False)
    fig.update_layout(title=f"Principal components of the minute features: PC1 {explained[0]:.0%}, "
                            f"PC2 {explained[1]:.0%} of the variance",
                      xaxis_title="PC1", yaxis_title="PC2", legend=dict(orientation="h", y=-0.15))
    show(fig, "pca", width=900, height=700)
    ''', slides=["73"], treatment="lab data: the slide's PCA cartoon replaced by PCA on the slice's "
                                 "minute features")
    explain(
        "Set selection and reduction side by side on the same ten features.",
        "Selection keeps a subset — here, drop one of each pair correlated beyond 0.9 in "
        "absolute Spearman correlation, a stated threshold. Reduction builds new columns — "
        "here, principal components. The trade is interpretability against compactness.",
        "Runs the correlation filter (keeping the earlier column of each pair) and counts the "
        "principal components needed for 90 per cent of the variance.",
        "On these weakly correlated features neither buys much compactness: the filter drops "
        "one column, and seven components are needed for 90 per cent. Selection keeps named "
        "columns a transport authority can justify; reduction explains none of them.")
    nb.figure("selection_vs_reduction", r'''
    absolute = spearman.abs()
    dropped = []
    for position, first in enumerate(feature_columns):
        for second in feature_columns[position + 1:]:
            if first not in dropped and second not in dropped and absolute.loc[first, second] > 0.9:
                dropped.append(second)
    kept = [c for c in feature_columns if c not in dropped]
    needed = int(np.searchsorted(np.cumsum(explained), 0.90) + 1)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], subplot_titles=(
        f"selection: |Spearman| > 0.9 drops {len(dropped)} of {len(feature_columns)}",
        f"reduction: {needed} components for 90% of the variance"))
    fig.add_bar(y=feature_columns, x=[1] * len(feature_columns), orientation="h", showlegend=False,
                marker_color=[GREY if c in dropped else BLUE for c in feature_columns],
                text=["dropped" if c in dropped else "kept" for c in feature_columns], textposition="inside",
                row=1, col=1)
    fig.add_bar(x=[f"PC{m}" for m in range(1, len(explained) + 1)], y=explained, marker_color=ORANGE,
                showlegend=False, row=1, col=2)
    fig.add_scatter(x=[f"PC{m}" for m in range(1, len(explained) + 1)], y=np.cumsum(explained),
                    mode="lines+markers", line=dict(color=NAVY), showlegend=False, row=1, col=2)
    fig.add_hline(y=0.9, line=dict(color=RED, dash="dash"), row=1, col=2)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_layout(title="Selection keeps named columns; reduction replaces them",
                      margin=dict(l=190, r=30, t=80, b=60))
    show(fig, "selection_vs_reduction", width=1100, height=520)
    print("  kept:", kept)
    print("  dropped:", dropped)
    ''', slides=["84"], treatment="lab data: the slide's generated poster replaced by a correlation "
                                 "filter and PCA on the slice's minute features")

    # --- slides 75-80: Fourier ------------------------------------------------------
    nb.md("### A second description: the spectrum\n\n*Slides: \"A spectrum is worth computing where "
          "the structure is concentrated in few rhythms\", \"Definition — the discrete Fourier "
          "transform of a window\", \"Adding independent errors multiplies their spectra, and "
          "leaves the normal curve\", and \"Feature extraction in timeseries\" (three slides on the "
          "transform).*")
    explain(
        "Write the transform and its inverse, and check numpy's against them.",
        "The transform rewrites N readings as N amplitudes, one per rhythm that fits a whole "
        "number of times into the window; the inverse gives the readings back, and energy "
        "is the same under both descriptions [@cooley1965; @oppenheim2010, ch. 8].",
        "Writes X_k and x_n in sympy; builds the transform as an N × N matrix for N = 4, "
        "applies it to 1, 2, 3, 4 and compares with `numpy.fft.fft`; checks the inverse and "
        "Parseval's identity.",
        "The formula on the slide is exactly what `numpy.fft` computes; no information is "
        "created or lost.")
    nb.equation("dft_pair", r'''
    n_, k_, N_ = sp.symbols("n k N", integer=True, nonnegative=True)
    xs, Xs = sp.IndexedBase("x"), sp.IndexedBase("X")
    formula(sp.Eq(Xs[k_], sp.Sum(xs[n_] * sp.exp(-2 * sp.pi * sp.I * k_ * n_ / N_), (n_, 0, N_ - 1)), evaluate=False),
            sp.Eq(xs[n_], sp.Sum(Xs[k_] * sp.exp(2 * sp.pi * sp.I * k_ * n_ / N_), (k_, 0, N_ - 1)) / N_, evaluate=False))
    size = 4
    matrix = sp.Matrix(size, size, lambda a, b: sp.exp(-2 * sp.pi * sp.I * a * b / size))
    sequence = sp.Matrix([1, 2, 3, 4])
    spectrum = sp.simplify(matrix * sequence)
    print("by the formula:", list(spectrum), "  numpy.fft:", np.fft.fft([1, 2, 3, 4]).round(10).tolist())
    print("inverse returns the readings:", list(sp.simplify(matrix.H * spectrum / size)))
    print("Parseval: sum |x|² =", sum(v**2 for v in sequence), "  (1/N) sum |X|² =",
          sp.simplify(sum(sp.Abs(v)**2 for v in spectrum) / size))
    ''', slides=["78"])
    explain(
        "State the two limits the sampling rate sets, with the slice's numbers.",
        "The finest resolvable rhythm is the sampling rate divided by the window length; "
        "nothing above half the sampling rate is observable [@shannon1949, Theorem 1]. "
        "Cutting a window out of a longer signal spreads each rhythm over its neighbours "
        "(spectral leakage), which has nothing to do with target leakage [@harris1978].",
        "Writes both limits in sympy and evaluates them at the vehicle's rate, 2 readings a "
        "second, for a five-minute window.",
        "One cycle per second is the ceiling, as the slide says: a fixed loop driven at "
        "walking pace has no rhythm up there worth a spectrum.")
    nb.equation("dft_limits", r'''
    fs, N = sp.symbols("f_s N", positive=True)
    formula(sp.Eq(sp.Symbol(r"\Delta f"), fs / N, evaluate=False), sp.Eq(sp.Symbol(r"f_{max}"), fs / 2, evaluate=False))
    rate = 1 / float(archive("bus_interval_s"))
    agrees("highest observable frequency at the vehicle's rate, cycles per second", rate / 2, 1, 2)
    print(f"resolution of a five-minute window ({int(300 * rate)} readings): {rate / (300 * rate):.4f} Hz, "
          "one cycle per 300 seconds")
    ''', slides=["76"])
    explain(
        "Draw the theorem of Block 1 in frequency, with the deck's own code.",
        "Adding independent errors combines their densities; in frequency that combination "
        "is a multiplication, so n additions raise one function to the n-th power, and "
        "everything away from zero frequency is crushed [@feller1971]. The normal curve is "
        "the shape that survives.",
        "Copies `figure_convolution()` from `make_figs.py` verbatim and calls it. It uses no "
        "data.",
        "Four additions already look normal, which is why the theorem helps at small n.")
    nb.source(FIGS, "figure_convolution")
    explain(
        "Draw the slide's figure.",
        "It is the deck's own code, with no data.",
        'Calls figure_convolution().',
        'Four additions, and the modulus of the transform crushed everywhere but near zero.')
    nb.figure("convolution", r'''
    figure_convolution()
    ''', slides=["77"], treatment="exact: the slide's own code")
    explain(
        "Build a square wave from sines: the time and frequency descriptions of one signal.",
        "The slide's illustration decomposes a square wave into sinusoids and shows their "
        "amplitudes along a frequency axis [@oppenheim2010, ch. 8].",
        "Sums the square wave's Fourier series, (4/π) Σ sin((2k − 1)t)/(2k − 1), with 1, 3 and "
        "15 terms, and draws the amplitudes 4/(π(2k − 1)) as a spectrum.",
        "Few terms already give the shape; the ripple at the jumps never disappears (Gibbs), "
        "which is spectral leakage's cousin: a sharp edge needs every rhythm.")
    nb.figure("square_wave", r'''
    t_grid = np.linspace(0, 4 * np.pi, 1200)
    square = np.sign(np.sin(t_grid))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=("time: partial sums of the series", "frequency: amplitude of each sine"))
    fig.add_scatter(x=t_grid, y=square, mode="lines", line=dict(color=GREY, width=1), name="square wave", row=1, col=1)
    for terms, colour in ((1, ORANGE), (3, GREEN), (15, RED)):
        partial = sum(4 / np.pi * np.sin((2 * m - 1) * t_grid) / (2 * m - 1) for m in range(1, terms + 1))
        fig.add_scatter(x=t_grid, y=partial, mode="lines", line=dict(color=colour, width=2),
                        name=f"{terms} term(s)", row=1, col=1)
    harmonics = np.arange(1, 16)
    fig.add_bar(x=2 * harmonics - 1, y=4 / (np.pi * (2 * harmonics - 1)), marker_color=BLUE,
                showlegend=False, row=1, col=2)
    fig.update_xaxes(title_text="t", row=1, col=1)
    fig.update_xaxes(title_text="frequency (multiples of the fundamental)", row=1, col=2)
    fig.update_layout(title="One signal, two descriptions", legend=dict(orientation="h", y=-0.2))
    show(fig, "square_wave", height=440)
    ''', slides=["78"], treatment="exact: closed form (the slide's diagram of a square wave and "
                                 "its sinusoids)")
    explain(
        "Rebuild one lap of the shuttle's route from its Fourier terms.",
        "The slide's animation draws a shape with rotating circles (epicycles): each circle "
        "is one term of the transform of a closed curve. The route is a closed curve, so "
        "its own lap can be drawn the same way.",
        "Takes the first lap of the loop on 23 January — the day the shuttle drove the loop; "
        "on 22 January it mostly shuttled along the southern road — from the first moving "
        "reading on the loop's north edge until the vehicle is back within 2 m of it after at "
        "least 150 m; resamples it to 256 points equally spaced along the path, writes each "
        "point as east + i·north, applies `numpy.fft.fft`, and rebuilds the lap from the 3, "
        "5, 11 and 21 lowest-frequency terms; the circles of the first four rotating terms are "
        "drawn, chained from the lap's centroid, at the lap's start.",
        "A handful of terms already gives the loop's shape; the corners need more, as the "
        "square wave's jumps did.")
    nb.figure("route_epicycles", r'''
    lap_day = in_time[in_time["_t"].dt.date.astype(str) == "2020-01-23"].reset_index(drop=True)
    east_all, north_all = to_metres(lap_day["lat"], lap_day["lon"])
    path = np.r_[0, np.cumsum(np.hypot(np.diff(east_all), np.diff(north_all)))]
    begin = int(np.where((north_all > 60) & (lap_day["speed"].abs().to_numpy() > 0.5))[0][0])
    after = np.arange(begin, len(lap_day))
    back = after[(path[after] - path[begin] > 150)
                 & (np.hypot(east_all[after] - east_all[begin], north_all[after] - north_all[begin]) < 2.0)]
    finish = int(back[0])
    arc = path[begin:finish + 1] - path[begin]
    even = np.linspace(0, arc[-1], 256, endpoint=False)
    z = np.interp(even, arc, east_all[begin:finish + 1]) + 1j * np.interp(even, arc, north_all[begin:finish + 1])
    Z = np.fft.fft(z) / len(z)
    frequencies = np.fft.fftfreq(len(z), d=1 / len(z)).astype(int)

    def rebuild(terms):
        keep = np.argsort(np.abs(frequencies))[:terms]
        phase = np.exp(2j * np.pi * np.outer(np.arange(len(z)), frequencies[keep]) / len(z))
        return phase @ Z[keep]

    fig = go.Figure()
    fig.add_scatter(x=z.real, y=z.imag, mode="lines", line=dict(color=GREY, width=6), name="the lap (slice)")
    for terms, colour in ((3, ORANGE), (5, GREEN), (11, BLUE), (21, RED)):
        curve = rebuild(terms)
        fig.add_scatter(x=np.r_[curve.real, curve.real[0]], y=np.r_[curve.imag, curve.imag[0]],
                        mode="lines", line=dict(color=colour, width=2), name=f"{terms} term(s)")
    order = np.argsort(np.abs(frequencies))[1:5]          # the four rotating terms after the centre
    centre = Z[0]                                         # the lap's centroid: the fixed pivot
    for term in order:
        radius = abs(Z[term])
        angle = np.linspace(0, 2 * np.pi, 100)
        fig.add_scatter(x=centre.real + radius * np.cos(angle), y=centre.imag + radius * np.sin(angle),
                        mode="lines", line=dict(color=NAVY, width=1), showlegend=False)
        tip = centre + Z[term]
        fig.add_scatter(x=[centre.real, tip.real], y=[centre.imag, tip.imag], mode="lines",
                        line=dict(color=NAVY, width=2), showlegend=False)
        centre = tip
    fig.update_layout(title=f"One lap ({arc[-1]:.0f} m, 23 January, {lap_day['_t'][begin]:%H:%M:%S} to "
                            f"{lap_day['_t'][finish]:%H:%M:%S} UTC) rebuilt from its Fourier terms",
                      xaxis_title="metres east", yaxis_title="metres north", yaxis_scaleanchor="x",
                      legend=dict(orientation="h", y=-0.15))
    show(fig, "route_epicycles", width=800, height=760)
    ''', slides=["79"], treatment="lab data: the slide's animated epicycles replaced by the Fourier "
                                 "terms of one lap of the slice's route, static")
    explain(
        "Compute the spectrum of the shuttle's speed, window by window, over a day.",
        "The slide shows a waterfall of spectra from an acoustic sensor, where the structure "
        "is in rhythm. The slice's speed can be put through the same machinery to see "
        "whether it has any.",
        "Puts 22 January's speed on a regular 0.5-second grid (mean per half-second, short "
        "gaps interpolated), cuts it into five-minute windows with at least 90 per cent of "
        "their readings (skipping windows at rest, whose spectrum is empty), and computes each window's periodogram with "
        "`scipy.signal.periodogram` (a Hann window against spectral leakage); draws them as a "
        "spectrogram.",
        "The power sits below a few hundredths of a cycle per second — the rhythm of stops "
        "and laps — and nothing lives near the one-cycle-per-second ceiling. The deck's "
        "verdict, measured: this series does not need a spectrum.")
    nb.figure("speed_spectrogram", r'''
    grid = slice_day_one.set_index("_t")["speed"].resample("500ms").mean()
    rows, starts = [], []
    for start, part in grid.groupby(grid.index.floor("5min")):
        if part.notna().mean() >= 0.9 and len(part) == 600 and part.std() > 0.01:
            frequencies, power = signal.periodogram(part.interpolate(limit_direction="both").to_numpy(),
                                                    fs=2.0, window="hann", detrend="constant")
            rows.append(power)
            starts.append(start)
    spectra = np.array(rows)
    fig = go.Figure(go.Heatmap(z=np.log10(spectra.T[1:] + 1e-6), x=[s.strftime("%H:%M") for s in starts],
                               y=frequencies[1:], colorscale="Viridis",
                               colorbar=dict(title="log₁₀ power")))
    fig.update_layout(title="Spectrogram of the shuttle's speed, five-minute windows, 22 January",
                      xaxis_title="window start (UTC)", yaxis_title="frequency, cycles per second")
    show(fig, "speed_spectrogram", height=500)
    share_low = spectra[:, (frequencies > 0) & (frequencies <= 0.05)].sum() / spectra[:, frequencies > 0].sum()
    print(f"  {len(starts)} windows; share of the power at or below 0.05 Hz: {share_low:.3f}")
    ''', slides=["80"], treatment="lab data: the slide's acoustic waterfall replaced by a spectrogram "
                                 "of the slice's speed")

    # --- slides 81-83: split and join ---------------------------------------------------
    nb.md("### The split that keeps time, and the price of a join\n\n*Slides: \"Definition — a "
          "split by time, never at random for timeseries\", \"Definition — a nearest match within a "
          "tolerance (fuzzy time-based join)\" and \"Widening the tolerance thirtyfold buys few rows "
          "and costs accuracy on all of them\".*")
    explain(
        "Write the split by time, and define it as the solution does.",
        "Consecutive readings correlate at 0.9967, so a random split scores the model on "
        "near-copies of its training rows [@roberts2017; @bergmeir2012; "
        "@huyen2022, ch. 4, \"Data Leakage\"]. Sorting is part of the operation.",
        "Writes the two sets as sympy condition sets on the time t, with t_c the time of "
        "the first test row: the row ⌊fn⌋ + 1 in sorted order, t↑ denoting the sorted "
        "times. Then writes the condition under which a cut *by row* gives those sets: the "
        "last training row must be strictly earlier than the first test row.",
        "A cut by row is the definition's split only when no instant straddles it. The "
        "next cells show that on the lab data one does.")
    nb.equation("split_by_time", r'''
    t_, t_c = sp.Symbol("t", real=True), sp.Symbol("t_c", real=True)
    f_, n_rows = sp.Symbol("f", positive=True), sp.Symbol("n", positive=True, integer=True)
    sorted_t = sp.IndexedBase(r"t^{\uparrow}")
    formula(sp.Eq(sp.Symbol(r"\mathrm{train}"), sp.ConditionSet(t_, sp.Lt(t_, t_c), sp.Reals), evaluate=False),
            sp.Eq(sp.Symbol(r"\mathrm{test}"), sp.ConditionSet(t_, sp.Ge(t_, t_c), sp.Reals), evaluate=False),
            sp.Eq(t_c, sorted_t[sp.floor(f_ * n_rows) + 1], evaluate=False))
    formula(sp.Symbol(r"\text{a cut by row gives these sets only if}"),
            sp.StrictLessThan(sorted_t[sp.floor(f_ * n_rows)], sorted_t[sp.floor(f_ * n_rows) + 1], evaluate=False))
    ''', slides=["81"])
    explain(
        "Define split_by_time as the Lab 4 solution writes it.",
        "It is the second deliverable of Laboratory 4.",
        "Copies split_by_time from solutions/lab_04.py: sort by timestamp_utc, then cut at "
        "row `int(len × fraction)`.",
        "The next cell tests it against the definition.")
    nb.source(f"{S}/lab_04.py", "split_by_time", cite="[@roberts2017; @bergmeir2012]")
    explain(
        "Test `split_by_time` against the definition on the first generated day.",
        "Twelve phones report at every instant of the generated day, so every timestamp is "
        "shared by twelve rows, and a cut by row can fall between two rows with the same "
        "time. The definition says train = {t < t_c}; the lab's check asks only that the "
        "last training time be at or before the first test time, which a straddled instant "
        "satisfies.",
        "Splits the shuffled day with `split_by_time(…, 0.7)` as the solution's "
        "demonstration does; prints the last training time and the first test time; counts "
        "the rows stamped t_c on each side; and evaluates the row-cut condition written "
        "above with sympy on the actual times.",
        "7,559 rows train, ending at 09:10:24.597, and 3,241 test, starting at 09:10:24.597: "
        "eleven readings of the instant t_c are in the training set and one in the test "
        "set. The lab's check passes (≤ holds) and the definition does not (< fails). The "
        "cause is one line of arithmetic: 0.7 × 10,800 is 7,559.999… in floating point, so "
        "`int()` gives 7,559 instead of the 7,560 = 630 × 12 that would have fallen between "
        "instants. The lab is delivered and is not changed; the fix, where it matters, is "
        "the one `assemble` makes below — cut at an instant, not at a row.")
    nb.code(r'''
    shuffled_day = phones.sample(frac=1.0, random_state=20200122).reset_index(drop=True)
    train_part, test_part = split_by_time(shuffled_day, 0.7)
    last_train, first_test = train_part["timestamp_utc"].max(), test_part["timestamp_utc"].min()
    print(f"train {len(train_part):,} rows, last time {last_train:%H:%M:%S.%f}; "
          f"test {len(test_part):,} rows, first time {first_test:%H:%M:%S.%f}")
    print(f"rows stamped t_c = {first_test:%H:%M:%S.%f}: {int((train_part['timestamp_utc'] == first_test).sum())} "
          f"in train, {int((test_part['timestamp_utc'] == first_test).sum())} in test")
    print("phones reporting at every instant:", sorted(phones.groupby("timestamp_utc").size().unique()))
    row_cut_condition = sp.StrictLessThan(sorted_t[sp.floor(f_ * n_rows)], sorted_t[sp.floor(f_ * n_rows) + 1])
    stamps_sorted = np.sort(phones["timestamp_utc"].astype("int64").to_numpy())
    row_cut = int(len(stamps_sorted) * 0.7)                        # the lab's own arithmetic
    print(f"0.7 × {len(phones):,} = {0.7 * len(phones)!r} → cut after row {row_cut:,}; "
          f"t↑[{row_cut}] < t↑[{row_cut + 1}]:",
          row_cut_condition.subs({sorted_t[sp.floor(f_ * n_rows)]: int(stamps_sorted[row_cut - 1]),
                                  sorted_t[sp.floor(f_ * n_rows) + 1]: int(stamps_sorted[row_cut])}))
    print("the lab's check, last training time ≤ first test time:", bool(last_train <= first_test),
          "  the definition, last training time < t_c:", bool(last_train < first_test))
    beside("split by time on the first generated day",
           f"last training time = first test time = {first_test:%H:%M:%S.%f}; "
           f"{int((train_part['timestamp_utc'] == first_test).sum())} rows of that instant train, "
           f"{int((test_part['timestamp_utc'] == first_test).sum())} test",
           "train = rows before the cut instant, test = rows after it (slide 81)",
           "split_by_time cuts at row int(0.7 × 10,800) = 7,559 (floating point), inside a group of "
           "twelve simultaneous rows; the lab's check tests ≤, so it passes. Lab 3's demonstration "
           "cuts the same way")
    ''')
    explain(
        "Write the nearest match within a tolerance, and define the function that measures it.",
        "Each phone row is paired with the vehicle reading nearest in time, if within τ "
        "seconds; widening τ buys matches and spends accuracy. The matched share must never "
        "fall as τ grows. The Lab 4 stub cites McKinney (2022) for this definition; the "
        "book's § 8.2 covers joins on equal keys (`pandas.merge`), not the nearest-in-time "
        "join the lab uses (`pandas.merge_asof`, documented by pandas itself), so no source "
        "is cited for it here.",
        "Writes the match as a sympy Piecewise and the share as a sum of indicators over "
        "the n_p phone rows; copies `join_growth` from `solutions/lab_04.py` (built on "
        "`pandas.merge_asof`).",
        "The next figures measure the trade: the rows it buys, and the accuracy it spends.")
    nb.equation("nearest_match", r'''
    t_b, tau = sp.Symbol("t_b", real=True), sp.Symbol(r"\tau", positive=True)
    p_ = sp.Symbol("p", positive=True, integer=True)
    t_phone = sp.IndexedBase("t")
    gap_p = sp.Abs(t_b - t_phone[p_])
    nearest = sp.Function(r"\min_b")(gap_p)
    match_rule = sp.Piecewise((sp.Function(r"\operatorname*{arg\,min}_b")(gap_p), sp.Le(nearest, tau)),
                         (sp.Symbol(r"\text{none}"), True))
    n_p = sp.Symbol("n_p", positive=True, integer=True)
    matched_share = sp.Sum(sp.Piecewise((1, sp.Le(nearest, tau)), (0, True)), (p_, 1, n_p)) / n_p
    formula(sp.Eq(sp.Function(r"\mathrm{match}")(p_), match_rule, evaluate=False))
    formula(sp.Eq(sp.Symbol(r"\text{matched share}"), matched_share, evaluate=False))
    ''', slides=["82"])
    explain(
        "Define join_growth as the Lab 4 solution writes it.",
        "It is the third deliverable of Laboratory 4.",
        "Copies join_growth from solutions/lab_04.py: both clocks forced to one resolution, then "
        "pandas.merge_asof with direction \"nearest\" at each tolerance.",
        "The next cells measure the trade with it.")
    nb.source(f"{S}/lab_04.py", "join_growth")
    explain(
        "Measure what widening the tolerance buys, and draw it with the deck's function.",
        "The slide measures the archive: 95.3 per cent of labelled phone rows matched at one "
        "second, 97.4 at thirty — 2.1 points for thirty times the tolerance. Against the "
        "shipped vehicle, which reports twice a second, every generated phone row matches at "
        "one second, so the lab's own recipe thins the vehicle to one reading every 30 "
        "seconds to have a trade to measure.",
        "Runs `join_growth` against the shipped vehicle and against the thinned one; copies "
        "`figure_join_cost()` from `make_figs.py` verbatim and feeds it the thinned rates; "
        "prints the archive's beside.",
        "The share never falls as the tolerance widens. The archive's trade (2.1 points) and "
        "the lab's (from under 10 to 100 per cent) differ because the lab's vehicle is thinned "
        "on purpose; the method is the one on the slide. The thinned curve shows the "
        "opposite of the slide's \"buys few rows\", and says nothing about accuracy; the "
        "cell after it measures both.")
    nb.source(FIGS, "figure_join_cost")
    explain(
        "Measure the join both ways and draw the thinned vehicle's trade.",
        'The shipped vehicle gives a flat curve; the thinned one gives a trade to see.',
        "Runs join_growth() twice, feeds figure_join_cost() the thinned shares, checks that the share never falls, and prints the archive's shares beside.",
        "The method is the slide's; the numbers are the lab data's.")
    nb.figure("join_cost", r'''
    dense = join_growth(phones, bus, TOLERANCES)
    thinned = join_growth(phones, in_time.iloc[::60].drop(columns="_t"), TOLERANCES)
    gain = figure_join_cost({"join_match_rate": {"value": {str(s): thinned[s] for s in TOLERANCES}}})
    print("  shipped vehicle, per cent matched:", dense)
    print("  vehicle thinned to one reading per 30 s:", thinned)
    print("  never falls as the tolerance widens:",
          all(a <= b for a, b in zip(list(thinned.values()), list(thinned.values())[1:])))
    beside("matched share from 1 s to 30 s, per cent", f"{thinned[1]} to {thinned[30]} (thinned vehicle)",
           f"{archive('join_match_rate')['1']} to {archive('join_match_rate')['30']} (archive)",
           "the archive's phones and vehicle were recorded independently; the lab's vehicle is thinned "
           "to make a trade measurable")
    beside("points bought by thirty times the tolerance", gain["points"],
           f"{archive('join_match_gain')['points']} (archive)", "as above")
    ''', slides=["83"], treatment="lab data: the slide's function fed the lab's thinned-vehicle "
                                 "match rates; archive rates printed beside")
    explain(
        "Measure what widening the tolerance costs in accuracy, and find a curve on the lab "
        "data that saturates as the archive's does.",
        "The slide's claim has two halves: thirty times the tolerance buys few rows, and it "
        "costs accuracy. The thinned vehicle cannot show the first half — thinning it evenly "
        "puts every phone row within 15 seconds of a kept reading, so the share climbs to "
        "100 per cent — and the share alone cannot show the second. Two measurements can. "
        "On the thinned vehicle, the shipped vehicle still knows the speed at every phone "
        "instant, so the error of each attached speed can be measured. And the shipped "
        "vehicle file has real outages — gaps of tens of seconds to a quarter of an hour "
        "when it did not report — so a clock ticking at the phones' rate through each "
        "service day, matched against the unthinned vehicle, shows how a real stream's "
        "share grows with the tolerance.",
        "For the thinned vehicle and each tolerance: the share of phone rows matched, the "
        "mean and largest |t_bus − t_phone| over the matched rows, and the mean |attached "
        "speed − speed at the phone's instant|, the latter read from the shipped vehicle "
        "(every phone instant has a shipped reading within half a second). It checks the "
        "shares against `join_growth`'s, and whether a row matched at 1 s keeps the same "
        "partner at 30 s. For the service-day clock (every 0.993 s from each day's first to "
        "its last vehicle reading): the share matched and |t_bus − t_phone|. Finally the "
        "price of a partner Δ seconds away: the mean |v(t + Δ) − v(t)| of the shipped "
        "vehicle's own speed, over every reading that has a partner Δ seconds later.",
        "Both halves of the slide hold on the lab data once they are measured. Against the "
        "real outages the curve saturates as the archive's does: about 91 per cent of the "
        "clock matches at 1 second and under 94 at 30 — under three points for thirty times "
        "the tolerance. And a partner up to 30 seconds away carries a speed that differs "
        "from the moment described by 0.75 m/s on average, against 0.06 at 1 second. One "
        "precision on the slide's wording: a nearest match never changes the partner of a "
        "row that was already matched, so widening does not make those rows worse; what it "
        "spends is the guarantee — at 30 seconds, any matched row may be 30 seconds off, "
        "and the table no longer says which.")
    nb.figure("join_accuracy", r'''
    stamp = "datetime64[ns, UTC]"
    vehicle_rows = (in_time[["_t", "speed"]].assign(_t=lambda frame: frame["_t"].astype(stamp))
                      .assign(vehicle_time=lambda frame: frame["_t"]))


    def nearest_vehicle(instants, vehicle, tolerance_s):
        """Each instant's nearest vehicle reading within the tolerance (pandas.merge_asof)."""
        left = pd.DataFrame({"_t": pd.to_datetime(pd.Series(instants), utc=True).astype(stamp)}).sort_values("_t")
        return pd.merge_asof(left, vehicle.sort_values("_t"), on="_t", direction="nearest",
                             tolerance=pd.Timedelta(seconds=tolerance_s)).reset_index(drop=True)


    at_the_moment = nearest_vehicle(phones["timestamp_utc"], vehicle_rows, 0.5)["speed"]
    assert at_the_moment.notna().all()
    thinned_rows = vehicle_rows.iloc[::60]
    cost, partners = [], {}
    for tolerance in TOLERANCES:
        matched = nearest_vehicle(phones["timestamp_utc"], thinned_rows, tolerance)
        found = matched["speed"].notna()
        apart = (matched["vehicle_time"] - matched["_t"]).dt.total_seconds().abs()
        partners[tolerance] = matched["vehicle_time"]
        cost.append({"tolerance (s)": tolerance, "matched %": round(100 * float(found.mean()), 1),
                     "mean |Δt| (s)": round(float(apart[found].mean()), 2),
                     "largest |Δt| (s)": round(float(apart[found].max()), 2),
                     "mean speed error (m/s)": round(float((matched["speed"] - at_the_moment)[found].abs().mean()), 3)})
    cost = pd.DataFrame(cost).set_index("tolerance (s)")
    assert cost["matched %"].to_dict() == {s: thinned[s] for s in TOLERANCES}, "differs from join_growth"
    kept = partners[1].notna()
    print("thinned vehicle, one reading per 30 s:")
    print(cost.to_string())
    print(f"  rows matched at 1 s whose partner is the same at 30 s: "
          f"{int((partners[1][kept] == partners[30][kept]).sum())} of {int(kept.sum())}")

    service_clock = pd.DatetimeIndex(np.concatenate([
        pd.date_range(part["_t"].min(), part["_t"].max(), freq="993ms").tz_convert("UTC").tz_localize(None).values
        for _, part in in_time.groupby(in_time["_t"].dt.date)])).tz_localize("UTC")
    outages = in_time.groupby(in_time["_t"].dt.date)["_t"].diff().dt.total_seconds()
    real = []
    for tolerance in TOLERANCES:
        matched = nearest_vehicle(service_clock, vehicle_rows, tolerance)
        found = matched["speed"].notna()
        apart = (matched["vehicle_time"] - matched["_t"]).dt.total_seconds().abs()
        real.append({"tolerance (s)": tolerance, "matched %": round(100 * float(found.mean()), 1),
                     "mean |Δt| (s)": round(float(apart[found].mean()), 3),
                     "matched rows more than 1 s off (%)": round(100 * float((apart[found] > 1).mean()), 2)})
    real = pd.DataFrame(real).set_index("tolerance (s)")
    print(f"\nservice-day clock, {len(service_clock):,} instants against the shipped vehicle; the vehicle's gaps "
          f"longer than 1 s: {int((outages > 1).sum())}, longer than 30 s: {int((outages > 30).sum())}, "
          f"longest {outages.max():.0f} s:")
    print(real.to_string())

    price = {}
    later = vehicle_rows[["_t", "speed"]].rename(columns={"speed": "speed_later"})
    for delta in TOLERANCES:
        ahead = vehicle_rows[["_t", "speed"]].assign(_t=lambda frame: frame["_t"] + pd.Timedelta(seconds=delta))
        paired = pd.merge_asof(ahead.sort_values("_t"), later.sort_values("_t"), on="_t", direction="nearest",
                               tolerance=pd.Timedelta(milliseconds=250)).dropna()
        price[delta] = float((paired["speed_later"] - paired["speed"]).abs().mean())
    print("\nmean |v(t + Δ) − v(t)| of the vehicle's own speed, m/s:",
          {delta: round(value, 3) for delta, value in price.items()})

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=(
        "rows bought: share of phone instants matched", "accuracy spent: speed error of the attached reading"))
    for label, shares, colour in (("vehicle as shipped, twice a second", dense, GREY),
                                  ("vehicle thinned to one reading per 30 s", thinned, ORANGE),
                                  ("phone-rate clock through each service day, real outages", real["matched %"].to_dict(), BLUE)):
        fig.add_scatter(x=list(TOLERANCES), y=[shares[s] for s in TOLERANCES], mode="lines+markers",
                        line=dict(color=colour), name=label, row=1, col=1)
    fig.add_scatter(x=list(TOLERANCES), y=cost["mean speed error (m/s)"], mode="lines+markers",
                    line=dict(color=ORANGE, dash="dot"), name="thinned vehicle: mean error on matched rows",
                    row=1, col=2)
    fig.add_scatter(x=list(TOLERANCES), y=[price[s] for s in TOLERANCES], mode="lines+markers",
                    line=dict(color=RED), name="a partner Δ s away: mean |v(t + Δ) − v(t)|", row=1, col=2)
    fig.update_xaxes(type="log", tickvals=list(TOLERANCES), title_text="tolerance, seconds")
    fig.update_yaxes(title_text="per cent matched", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="m/s", row=1, col=2)
    fig.update_layout(title="What widening the tolerance buys, and what it costs, on the lab data",
                      legend=dict(orientation="h", y=-0.25), margin=dict(l=70, r=30, t=80, b=150))
    show(fig, "join_accuracy", width=1100, height=540)
    beside("matched share from 1 s to 30 s against a stream with real outages, per cent",
           f"{real.loc[1, 'matched %']} to {real.loc[30, 'matched %']} (phone-rate clock, shipped vehicle)",
           f"{archive('join_match_rate')['1']} to {archive('join_match_rate')['30']} (archive)",
           "the clock is synthetic (the phones' rate, both service days); the outages are the vehicle's own")
    ''', slides=["83"], treatment="lab data: the accuracy the slide says widening costs, measured "
                                 "on the thinned vehicle and on the shipped vehicle's real outages")

    # --- slide 85: the hand-off ------------------------------------------------------------
    nb.md("### The table this module hands on\n\n*Slides: \"Definition — the table this module hands "
          "to the next steps\" and \"Build dataset\".* The pipeline slide's point — fit the "
          "transform and the model as one object, so that a live request is prepared as the "
          "training rows were — is the hand-off's manifest.")
    explain(
        "Write the hand-off's clauses.",
        "One row per phone per window; a mask beside every filled column and no other; the "
        "split recorded as an instant at the start of a window, so no window straddles it; "
        "the transform and the ledger stored unchanged [@huyen2022; @kuhn2019, ch. 8].",
        "Writes the clauses in sympy — the key's count equal to the row count, the mask as "
        "a Piecewise, the cut instant as the window of row ⌊fn⌋ + 1 in (window, phone) "
        "order, and train and test as condition sets on the window; copies `assemble` "
        "from `solutions/lab_04.py`.",
        "The next cell builds the table and checks each clause.")
    nb.equation("handoff", r'''
    n_rows, f_ = sp.Symbol("n", positive=True, integer=True), sp.Symbol("f", positive=True)
    w_, t_c = sp.Symbol("w", real=True), sp.Symbol("t_c", real=True)
    ordered_windows = sp.IndexedBase(r"\mathrm{window}")
    formula(sp.Eq(sp.Symbol(r"|\{(\mathrm{phone\_id}, \mathrm{window})\}|"), n_rows, evaluate=False),
            sp.Eq(sp.Symbol(r"\mathrm{mask}_c"),
                  sp.Piecewise((1, sp.Symbol(r"x_c\ \text{absent}")), (0, True)), evaluate=False),
            sp.Symbol(r"\text{beside every filled } c"))
    formula(sp.Eq(t_c, ordered_windows[sp.floor(f_ * n_rows) + 1], evaluate=False),
            sp.Eq(sp.Symbol(r"\mathrm{train}"), sp.ConditionSet(w_, sp.Lt(w_, t_c), sp.Reals), evaluate=False),
            sp.Eq(sp.Symbol(r"\mathrm{test}"), sp.ConditionSet(w_, sp.Ge(w_, t_c), sp.Reals), evaluate=False))
    ''', slides=["85"])
    explain(
        "Define assemble as the Lab 4 solution writes it.",
        "It is the fourth deliverable of Laboratory 4, and the object Modules 3, 4 and 5 open.",
        "Copies assemble from solutions/lab_04.py: fill and scale with the stored constants, a "
        "mask beside what was filled, the cut moved to the start of a window, and the manifest.",
        "The next cell builds the hand-off with it.")
    nb.source(f"{S}/lab_04.py", "assemble", cite="[@huyen2022; @kuhn2019, ch. 8]")
    explain(
        "Build the hand-off from the three labs' functions and draw its split.",
        "The slide's picture is a generated drawing of the cut; the manifest records the "
        "real one.",
        "Aligns the generated day with the slice (Lab 1); finds the cut instant as "
        "`assemble` will — the window of row ⌊0.7 n⌋ + 1 in (window, phone) order — and "
        "fits the transform (Lab 3) on the rows before it, the rows `assemble` will label "
        "train; assembles the table (Lab 4), checks that its split point is that instant, "
        "and draws the rows per window with the split point. Checks that no window has rows "
        "on both sides and that a mask exists exactly for the columns with gaps. Then fits "
        "the transform the way the Lab 4 solution's demonstration does — on the first "
        "`int(0.7 n)` rows in window order, before `assemble` has moved the cut back to the "
        "start of a window — and measures how much that changes the stored constants and "
        "the table.",
        "The split point is an instant, recorded as text; the counts beside it add up. The "
        "solution's own demonstration fits θ on 1,503 rows while `assemble` trains on "
        "1,500, so three rows that end up in the test set help set the constants — against "
        "the slides' rule that the constants are fitted on training rows only. The effect is "
        "measurable and small: no constant moves by more than a thousandth of its unit, and "
        "no scaled value by more than a hundredth. This notebook's own table is fitted on the "
        "1,500 training rows; the lab is not changed.")
    nb.figure("split_timeline", r'''
    aligned, ledger = align(bus, phones, 5)
    stored_columns = ["phone_speed", "rssi1", "rssi2", "bus_speed"]
    by_window = aligned.sort_values(["window", "phone_id"]).reset_index(drop=True)
    cut_instant = by_window["window"].iloc[int(len(by_window) * TRAIN_FRACTION)]
    training = by_window[by_window["window"] < cut_instant]              # the rows assemble() will call train
    fitted = fit_preprocessing(training[stored_columns])
    handoff, manifest = assemble(aligned, fitted, ledger, TARGET_COLUMN, TRAIN_FRACTION)
    assert str(cut_instant) == manifest["split_point"] and len(training) == manifest["train_rows"]
    per_window = handoff.groupby(["window", "split"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for split, colour in (("train", BLUE), ("test", GREY)):
        if split in per_window:
            fig.add_bar(x=per_window.index, y=per_window[split], marker_color=colour, name=split)
    fig.add_vline(x=pd.Timestamp(manifest["split_point"]), line=dict(color=ORANGE, width=3))
    fig.update_layout(barmode="stack", title=f"The hand-off's split: t_c = {manifest['split_point']}, "
                                             f"{manifest['train_rows']:,} train and {manifest['test_rows']:,} test rows",
                      xaxis_title="five-second window (UTC)", yaxis_title="rows (phones) in the window",
                      legend=dict(orientation="h", y=-0.2))
    show(fig, "split_timeline", height=440)
    straddling = int((handoff.groupby("window")["split"].nunique() > 1).sum())
    with_gaps = sorted(c + "_missing" for c in stored_columns if aligned[c].isna().any())
    print(f"  rows {manifest['rows']:,}, columns {manifest['columns']}, key {manifest['key']}")
    print(f"  windows straddling the split: {straddling}")
    print(f"  masks written: {manifest['mask_columns']}; columns with gaps: {with_gaps}")
    print(f"  train + test = rows: {manifest['train_rows'] + manifest['test_rows'] == manifest['rows']}")

    # The Lab 4 solution's demonstration fits on the first int(0.7 n) rows in window order.
    solution_rows = aligned.sort_values("window").iloc[:int(len(aligned) * TRAIN_FRACTION)]
    solution_fitted = fit_preprocessing(solution_rows[stored_columns])
    solution_table, _ = assemble(aligned, solution_fitted, ledger, TARGET_COLUMN, TRAIN_FRACTION)
    print(f"\n  the solution fits θ on {len(solution_rows):,} rows; assemble() trains on "
          f"{manifest['train_rows']:,}; rows of the cut window among the fitted: "
          f"{int((solution_rows['window'] >= cut_instant).sum())}")
    moved = pd.DataFrame({part: {c: solution_fitted[part][c] - fitted[part][c] for c in stored_columns}
                          for part in ("medians", "means", "stds")})
    print("  θ(solution's rows) − θ(training rows):")
    print(moved.to_string(float_format=lambda value: f"{value:+.6f}"))
    largest_constant = float(moved.abs().max().max())
    largest_scaled = float((solution_table[stored_columns] - handoff[stored_columns]).abs().max().max())
    print(f"  largest change in a constant: {largest_constant:.5f}; in a scaled value of the table: {largest_scaled:.4f}")
    beside("rows the hand-off transform θ is fitted on", f"{len(training):,} (this notebook: the training rows)",
           f"{len(solution_rows):,} (the Lab 4 solution's demonstration)",
           f"the solution fits on the first int(0.7 n) rows before assemble() moves the cut back to the start "
           f"of a window, so {int((solution_rows['window'] >= cut_instant).sum())} test rows inform θ; no constant "
           f"moves by more than {largest_constant:.4f} and no scaled value by more than {largest_scaled:.3f}")
    ''', slides=["81"], treatment="lab data: the slide's generated drawing of the cut replaced by the "
                                 "hand-off's own split, from assemble()")

    # --- Lab 4 -------------------------------------------------------------------------
    nb.md("""
    ## Laboratory 4 — Windows, splits and what they cost

    *Slide: "Lab 4 — Windows, splits and what they cost".* Four functions,
    twenty-five minutes. The check compares your features with a reference, refuses a
    split whose halves overlap in time, requires the matched share to rise with the
    tolerance, and grades the hand-off's three clauses.
    """)
    nb.statement(f"{LABS}/04_windows_and_cost.py")
    nb.md("""
    **The stub's slide titles, against the deck.** The stub places the lab under "What
    the join costs"; no slide has that title. The slide that measures the join is
    "Widening the tolerance thirtyfold buys few rows and costs accuracy on all of them".
    Four of the definition titles are shortened or reworded in the stub: the deck's
    slides are "Definition — six window features", "Definition — a split by time, never
    at random for timeseries", "Definition — a nearest match within a tolerance (fuzzy
    time-based join)" and "Definition — the table this module hands to the next steps".
    "Definition — lag-k sample autocorrelation" is a hidden slide: it is in the deck
    file but not shown in the lecture, and the stub states the definition in full. The
    stub cites McKinney (2022) for the nearest match; the book covers joins on equal keys,
    not `merge_asof`.
    """)
    nb.md("### The solution\n\nThe four functions and their helper were defined above, where the "
          "deck introduces each concept. First, the lab file's own demonstration.")
    explain(
        "Run the stub's own demonstration against the solved functions.",
        "It is what a student sees when the file is complete.",
        "The stub's `__main__` block, verbatim.",
        "Six features of the last 60 phone speeds, a split by time, and the join's shares.")
    nb.step(f"{LABS}/04_windows_and_cost.py", 0)
    explain(
        "Let the solution's demonstration find its sibling labs and write its hand-off inside "
        "the working copy.",
        "The Lab 4 solution imports `align` and `fit_preprocessing` from `lab_01` and "
        "`lab_03`, and writes to `out/handoff/` next to `solutions/`.",
        "Points `__file__` at `solutions/lab_04.py` in the working copy. `lab_01` and "
        "`lab_03` are already modules holding the functions defined above, so the imports "
        "resolve to them, not to the files.",
        "The hand-off lands in the temporary copy's `out/handoff/`, never in the student's "
        "folder.")
    nb.code(r'''
    __file__ = str(Path.cwd() / "solutions" / "lab_04.py")
    ''')
    nb.md("""
    ### The solution's demonstration, step by step

    What `python3 solutions/lab_04.py` prints, one paragraph of its `__main__` block at a
    time, each verbatim. It ends the module: the object Modules 3, 4 and 5 open.
    """)
    lab4 = f"{S}/lab_04.py"
    explain(
        "Start the demonstration's narrator.",
        "One logger carries every line the solution prints, in the terminal and here.",
        "Creates `say` for Lab 4 and prints the lab's one-line summary.",
        "The seconds at the start of each line are the only output that differs between runs.")
    nb.paragraph(lab4, 1)
    explain(
        "Load both files and summarise the last minute of phone speed.",
        "The first deliverable: six features over exactly the last `width` readings, the "
        "autocorrelation by the Box–Jenkins estimator [@box2015, § 2.1.4].",
        "Loads the phones and the vehicle; computes `window_features` on the last 60 phone "
        "speeds; prints them, and the lag-1 autocorrelation pandas' `autocorr` gives on the "
        "same window.",
        "The two autocorrelations differ on the same sixty numbers: two estimators, and the "
        "course grades one.")
    nb.paragraph(lab4, 2)
    explain(
        "Split a shuffled copy of the day by time.",
        "The second deliverable must sort before it cuts; a shuffled table proves it does.",
        "Shuffles the phones (seed 20200122), calls `split_by_time(…, 0.7)`, and prints "
        "the sizes, the last training time, the first test time, and whether the first is "
        "at or before the second.",
        "\"No overlap: True\" — by the check's test (≤). The times printed are equal, "
        "09:10:24 on both sides, because this is the same straddled instant the cells after "
        "`split_by_time`'s definition counted: eleven readings of it train and one tests.")
    nb.paragraph(lab4, 3)
    explain(
        "Measure the join against the shipped vehicle and against the thinned one.",
        "The third deliverable: the matched share at each tolerance, which must never fall "
        "as the tolerance widens.",
        "Runs `join_growth` against the vehicle as shipped and thinned to one reading per "
        "30 seconds, and tabulates both.",
        "Flat at 100 per cent against the shipped vehicle; from 6.8 to 100 against the "
        "thinned one. What the thinned curve does not show — the accuracy spent, and a curve "
        "that saturates — was measured on the lab data above.")
    nb.paragraph(lab4, 4)
    explain(
        "Draw the two curves.",
        "The lab's own version of the slide's figure.",
        "Both shares against the tolerance on a logarithmic axis; `save_figure` shows it here.",
        "One flat line and one that climbs.")
    nb.paragraph(lab4, 5)
    explain(
        "Import the two sibling labs the hand-off is built from.",
        "The hand-off joins the module's labs: Lab 1's alignment and ledger, Lab 3's fitted "
        "transform.",
        "Puts the solution's folder on the import path and imports `align` and "
        "`fit_preprocessing`. In this notebook `lab_01` and `lab_03` are the modules "
        "registered above, so the imports return the functions on this page.",
        "Nothing is read from the files themselves.")
    nb.paragraph(lab4, 6)
    explain(
        "Align, fit the transform, and assemble the hand-off.",
        "The fourth deliverable: one table, its masks, its split, its transform and its "
        "ledger.",
        "Aligns the day, sorts it by window, fits `fit_preprocessing` on the first "
        "`int(len × 0.7)` rows — 1,503 — and calls `assemble`, which moves the cut back to "
        "the start of the window that row falls in and labels 1,500 rows train.",
        "The table is the one the split figure above drew. Its constants were fitted on 1,503 "
        "rows, three of which `assemble` then puts in the test set — the slip measured "
        "beside that figure, where no constant moved by more than a thousandth of its unit.")
    nb.paragraph(lab4, 7)
    explain(
        "Write the hand-off to `out/handoff/`.",
        "A hand-off is a file Module 3 can open, not a variable in a session.",
        "Writes the table as Parquet and the manifest (split point, transform, ledger) as "
        "JSON under the working copy's `out/handoff/`, and shows the first rows.",
        "2,148 rows, masks on the two beacon columns, 1,500 training rows and 648 test rows.")
    nb.paragraph(lab4, 8)
    explain(
        "Print what the check grades.",
        "The demonstration ends by telling the student what `verify/check_04.py` will ask.",
        "One narrated line.",
        "Features, split, join and the hand-off's three clauses — the split graded by ≤, "
        "which is why the straddled instant passes.")
    nb.paragraph(lab4, 9)
    explain(
        "Read back what the demonstration wrote, and restore `__file__`.",
        "A hand-off is only real if it can be opened.",
        "Reads `out/handoff/manifest.json` and the table from the working copy.",
        "The manifest carries the split point, the transform and the ledger — the four "
        "objects the closing slide lists.")
    nb.code(r'''
    __file__ = str(Path.cwd() / "lab_support.py")
    written = json.loads((Path.cwd() / "out" / "handoff" / "manifest.json").read_text())
    table_back = pd.read_parquet(Path.cwd() / "out" / "handoff" / "table.parquet")
    print("manifest keys:", sorted(written))
    print("table read back:", table_back.shape, "  split point:", written["split_point"])
    assert (written["rows"], written["train_rows"], written["test_rows"]) == (
        manifest["rows"], manifest["train_rows"], manifest["test_rows"]), "the two hand-offs differ"
    print("the file on disk and the table built in the split figure agree:",
          written["rows"], "rows,", written["train_rows"], "train,", written["test_rows"], "test")
    ''')
