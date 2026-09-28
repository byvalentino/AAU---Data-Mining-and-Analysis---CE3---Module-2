"""Block 1 — know the distribution, then clean — and Laboratory 1: distributions,
the central limit theorem, the six faults, the grain, the ledger, the upstream
profile, Simpson's paradox, and the Lab 1 solution."""

EX = "Module 2/exercises"
S = f"{EX}/solutions"
LABS = f"{EX}/labs"
FIGS = "Module 2/slides/make_figs.py"


def build(nb, explain) -> None:
    nb.md("""
    ---
    # Block 1 — Know the distribution, then clean

    **The question of this block:** what does the shape of a column allow you to do
    to it, and how do two files on two clocks become one table without losing a row
    nobody counted?

    *Slide: "The distribution of a column decides which rule may be applied to it".*
    The shape decides the outlier rule, the fill, the scaler and the model family. A
    z-score assumes a roughly symmetric, light-tailed column; the interquartile range
    does not. The one response to an outlier most often skipped is to investigate it.
    """)

    # --- slides 7 and 13: heavy tails and the outlier rule ------------------------------
    explain(
        "Show two columns of the slice whose shape rules out the z-score.",
        "The overview slide shows a heavily skewed histogram with one value far to the "
        "right. The slice has one of its own: `mileage`, which Module 1 found reaching "
        "65,535 — two to the sixteenth minus one, a 16-bit counter at its ceiling, not a "
        "long journey. `payload` is skewed without being broken.",
        "Draws both histograms (mileage on a logarithmic count axis, so the lone value "
        "shows) and marks the maximum.",
        "One reading of 65,535 among values from 0 to 126 is not an outlier to trim; it is "
        "a fault to investigate and record.")
    nb.figure("heavy_tails", r'''
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "mileage: one reading at the counter's ceiling", "payload: skewed, and real"))
    fig.add_histogram(x=bus["mileage"], nbinsx=120, marker_color=BLUE, showlegend=False, row=1, col=1)
    fig.add_histogram(x=bus["payload"], nbinsx=60, marker_color=BLUE, showlegend=False, row=1, col=2)
    fig.update_yaxes(type="log", title_text="readings (log scale)", row=1, col=1)
    fig.update_yaxes(title_text="readings", row=1, col=2)
    fig.update_xaxes(title_text="mileage (as recorded)", row=1, col=1)
    fig.update_xaxes(title_text="payload, kg", row=1, col=2)
    fig.add_annotation(x=65535, y=0.3, text="65,535 = 2¹⁶ − 1", showarrow=True, ay=-60, row=1, col=1)
    fig.update_layout(title="Two columns of the slice that a z-score would misread")
    show(fig, "heavy_tails", height=440)
    agrees("mileage maximum", bus["mileage"].max(), module1("mileage_max"), 0)
    agrees("2 to the 16th minus 1", 2 ** 16 - 1, module1("mileage_max"), 0)
    print(f"  readings at 65,535: {(bus['mileage'] == 65535).sum()}; the next largest value: "
          f"{bus.loc[bus['mileage'] < 65535, 'mileage'].max()}")
    ''', slides=["7"], treatment="lab data: the slide's insurance-claims histogram replaced by "
                                "the slice's mileage (one broken-counter reading) and payload")
    explain(
        "Count what the two outlier rules flag on four columns.",
        "The slide says signal strength is neither symmetric nor light-tailed, so the "
        "interquartile range is the right instrument. The two rules can be compared on the "
        "same columns.",
        "For payload, speed and mileage on the slice, and the heard `rssi1` readings of the "
        "generated phones, counts the rows beyond three standard deviations (the z-score "
        "rule) and beyond 1.5 interquartile ranges outside the middle half (Tukey's rule), "
        "beside each column's skewness, excess kurtosis (0 for the normal curve; a heavy "
        "tail makes it large) and interquartile range.",
        "The rules disagree by orders of magnitude, and the table says why each time. "
        "Tukey's 14,837 speed readings are not outliers: more than half the readings are at "
        "rest, both quartiles are 0, the interquartile range is 0, and every moving reading "
        "falls outside a band of width nought — the rule is undefined on this column, not "
        "strict. The generated `rssi1` is mildly skewed (0.56) and not heavy-tailed, and "
        "there the z-score flags more than Tukey does (8 against 4); the slide's sentence "
        "about signal strength describes the archive, which this notebook cannot open, and "
        "the generated readings do not reproduce it. Which rule is right is a statement "
        "about the shape, and it has to be measured first.")
    nb.code(r'''
    def outlier_counts(values):
        x = pd.Series(values).dropna().astype(float)
        z = (x - x.mean()) / x.std(ddof=0)
        q1, q3 = x.quantile([0.25, 0.75])
        spread = q3 - q1
        return {"rows": len(x), "skewness": round(float(stats.skew(x)), 2),
                "excess kurtosis": round(float(stats.kurtosis(x)), 2), "IQR": round(float(spread), 3),
                "beyond 3 sd (z-score)": int((z.abs() > 3).sum()),
                "beyond 1.5 IQR (Tukey)": int(((x < q1 - 1.5 * spread) | (x > q3 + 1.5 * spread)).sum())}

    table = pd.DataFrame({"payload (slice)": outlier_counts(bus["payload"]),
                          "speed (slice)": outlier_counts(bus["speed"]),
                          "mileage (slice)": outlier_counts(bus["mileage"]),
                          "rssi1 heard (generated)": outlier_counts(phones["rssi1"])}).T
    print(table.to_string())
    print(f"  speed: {100 * float((bus['speed'] == 0).mean()):.1f} per cent of readings are exactly 0, so "
          f"both quartiles are 0; Tukey flags every reading that is not 0: "
          f"{int((bus['speed'] != 0).sum()):,}")
    rssi1_row = table.loc["rssi1 heard (generated)"]
    beside("rssi1 heard: shape, and what the two rules flag",
           f"skewness {rssi1_row['skewness']}, excess kurtosis {rssi1_row['excess kurtosis']}; "
           f"z-score flags {int(rssi1_row['beyond 3 sd (z-score)'])}, Tukey {int(rssi1_row['beyond 1.5 IQR (Tukey)'])} (generated)",
           "\"Signal strength here is neither [symmetric nor light-tailed], so the interquartile "
           "range is the correct instrument\" (slide 13)",
           "the slide describes the archive's signal strength, which this notebook does not open; "
           "the generator draws each heard reading from a log-distance law plus 2 dB of normal "
           "noise, which is mildly skewed and light-tailed, so on the lab data the rules nearly agree")
    ''')

    # --- slides 15-18: the catalogue ---------------------------------------------------
    nb.md("""
    ### A catalogue of distributions

    *Slides: "Why do I need to know the underlying distribution of my data?" (four
    slides: three tables and the list of reasons).* The tables name eighteen families
    and what each describes. The slides' pictures are samples of 1,000 draws; they are
    redrawn below from numpy with the parameters printed in each panel title, with the
    exact density or mass function laid over each. The slides do not state their
    parameters, so these panels are illustrative: the shapes, not the values, are the
    point.
    """)
    explain(
        "Draw the first six families: normal, log-normal, uniform, exponential, Poisson, "
        "binomial.",
        "Knowing the family tells you the outlier rule, the fill and the scaler before any "
        "of them is chosen. Among these six, only the normal and the uniform are symmetric.",
        "Defines `catalogue()`, which draws 1,000 values per family with the course seed, "
        "plots a density histogram, and lays the exact curve (continuous) or mass "
        "(discrete) from `scipy.stats` over it.",
        "The exact curve and the draws agree; that agreement is what \"the data follow a "
        "normal distribution\" should mean, and it can be checked.")
    nb.figure("distributions_one", r'''
    def catalogue(families, title, name):
        rng = np.random.default_rng(20200122)
        fig = make_subplots(rows=2, cols=3, subplot_titles=[f[0] for f in families],
                            vertical_spacing=0.14, horizontal_spacing=0.07)
        for number, (label, law, discrete, window) in enumerate(families):
            row, col = number // 3 + 1, number % 3 + 1
            draws = law.rvs(size=1000, random_state=rng)
            shown = draws[(draws >= window[0]) & (draws <= window[1])]
            if discrete:
                values, counts = np.unique(shown, return_counts=True)
                fig.add_bar(x=values, y=counts / len(draws), marker_color=BLUE, opacity=0.7,
                            showlegend=False, row=row, col=col)
                support = np.arange(int(window[0]), int(window[1]) + 1)
                fig.add_scatter(x=support, y=law.pmf(support), mode="markers",
                                marker=dict(color=RED, size=6), showlegend=False, row=row, col=col)
            else:
                fig.add_histogram(x=shown, histnorm="probability density", nbinsx=30,
                                  marker_color=BLUE, opacity=0.7, showlegend=False, row=row, col=col)
                grid = np.linspace(window[0], window[1], 300)
                fig.add_scatter(x=grid, y=law.pdf(grid), mode="lines", line=dict(color=RED, width=2),
                                showlegend=False, row=row, col=col)
        fig.update_layout(title=title + " — 1,000 draws each (bars), exact law (red)")
        show(fig, name, height=620)

    catalogue([
        ("Normal, μ = 0, σ = 1", stats.norm(0, 1), False, (-4, 4)),
        ("Log-normal, σ = 1", stats.lognorm(1), False, (0, 15)),
        ("Uniform on [0, 1]", stats.uniform(0, 1), False, (0, 1)),
        ("Exponential, rate 1", stats.expon(), False, (0, 8)),
        ("Poisson, mean 5", stats.poisson(5), True, (0, 15)),
        ("Binomial, n = 10, p = 0.5", stats.binom(10, 0.5), True, (0, 10)),
    ], "Six families of the first table", "distributions_one")
    ''', slides=["15", "17", "18"], treatment="illustrative: numpy draws with stated "
                                             "parameters and the exact law laid over them")
    explain(
        "Draw the second six: Bernoulli, beta, gamma, Weibull, chi-square, Student's t.",
        "These are the families of proportions (beta), positive skewed waiting times "
        "(gamma, Weibull), and test statistics (chi-square, t).",
        "Calls `catalogue()` again.",
        "Four of the six are bounded or positive; a z-score rule on any of them flags "
        "only one tail.")
    nb.figure("distributions_two", r'''
    catalogue([
        ("Bernoulli, p = 0.5", stats.bernoulli(0.5), True, (0, 1)),
        ("Beta, a = 2, b = 5", stats.beta(2, 5), False, (0, 1)),
        ("Gamma, shape 2", stats.gamma(2), False, (0, 10)),
        ("Weibull, shape 2", stats.weibull_min(2), False, (0, 3)),
        ("Chi-square, k = 2", stats.chi2(2), False, (0, 12)),
        ("Student's t, ν = 10", stats.t(10), False, (-5, 5)),
    ], "Six families of the second table", "distributions_two")
    ''', slides=["16", "18"], treatment="illustrative: numpy draws with stated parameters "
                                       "and the exact law laid over them")
    explain(
        "Draw the last six: F, geometric, negative binomial, Cauchy, Pareto, Rayleigh.",
        "The Cauchy has no mean at all, so the central limit theorem never rescues its "
        "averages; the Pareto is the power law behind \"a few values carry most of the "
        "total\"; the Rayleigh is the magnitude of a two-dimensional normal error, which "
        "is how a position fix scatters.",
        "Calls `catalogue()` again; the Cauchy panel shows only the draws between −10 "
        "and 10, and the cell counts the ones outside.",
        "A family with an undefined mean is not a curiosity: any rule that averages it "
        "is unstable however much data arrives.")
    nb.figure("distributions_three", r'''
    catalogue([
        ("F, d₁ = 5, d₂ = 20", stats.f(5, 20), False, (0, 6)),
        ("Geometric, p = 0.5", stats.geom(0.5), True, (1, 10)),
        ("Negative binomial, r = 5, p = 0.5", stats.nbinom(5, 0.5), True, (0, 20)),
        ("Cauchy (draws between −10 and 10)", stats.cauchy(), False, (-10, 10)),
        ("Pareto, shape 3", stats.pareto(3), False, (1, 6)),
        ("Rayleigh, scale 1", stats.rayleigh(), False, (0, 4)),
    ], "Six families of the third table", "distributions_three")
    cauchy = stats.cauchy().rvs(size=1000, random_state=np.random.default_rng(20200122))
    print(f"  Cauchy draws outside [-10, 10]: {int((np.abs(cauchy) > 10).sum())} of 1,000; "
          f"largest magnitude {np.abs(cauchy).max():,.0f}")
    ''', slides=["17", "18"], treatment="illustrative: numpy draws with stated parameters "
                                       "and the exact law laid over them")

    # --- slides 14 and 19: the central limit theorem --------------------------------
    nb.md("### The central limit theorem, and what it does not say\n\n"
          "*Slides: \"Definition — the central limit theorem, and what it does not say\" and "
          "\"Central limit theorem\".*")
    explain(
        "Write the theorem as the definition slide states it.",
        "The standardised mean of n independent draws from one distribution with finite "
        "variance tends to the standard normal [@casella2002, § 5.5; @wasserman2004, § 5.4; "
        "@feller1971].",
        "Builds the standardised mean in sympy and states the limit; then checks the "
        "variance of a mean of n independent draws, σ²/n, by summing symbolically.",
        "The √n in the denominator is where independence enters: with correlated readings "
        "the variance of the mean is not σ²/n, and the theorem's n is the wrong n.")
    nb.equation("central_limit", r'''
    n = sp.Symbol("n", positive=True, integer=True)
    mu, sigma = sp.symbols(r"\mu \sigma", positive=True)
    xbar = sp.Symbol(r"\bar{x}")
    formula(sp.Eq(sp.Symbol("Z_n"), (xbar - mu) / (sigma / sp.sqrt(n)), evaluate=False),
            r"\xrightarrow[n \longrightarrow \infty]{\;d\;}", sp.Function(r"\mathcal{N}")(0, 1))
    formula(sp.Symbol(r"\text{independent draws, one distribution}"), sp.Lt(sigma**2, sp.oo, evaluate=False))
    i = sp.Symbol("i", integer=True)
    variance_of_mean = sp.Sum(sigma**2, (i, 1, n)).doit() / n**2      # independent: covariances vanish
    print("Var(mean of n independent draws) =", sp.simplify(variance_of_mean))
    ''', slides=["14", "19"])
    explain(
        "Redo the slide's demonstration: three populations, and the means of samples of 5 "
        "and of 30.",
        "The slide's grid shows an exponential, a uniform and a normal population and the "
        "distributions of their sample means [@casella2002, § 5.5].",
        "Draws 10,000 values from each population (exponential with mean 2, uniform on "
        "[0, 10], normal with mean 5 and standard deviation 2, as read off the slide's "
        "axes), then 1,000 sample means at n = 5 and n = 30, with the course seed.",
        "At n = 30 all three are bell-shaped, the exponential included. The picture is the "
        "slide's; the parameters are read from its axes, so it is labelled illustrative.")
    nb.figure("clt_grid", r'''
    rng = np.random.default_rng(20200122)
    populations = [("Exponential, mean 2", lambda size: rng.exponential(2.0, size)),
                   ("Uniform on [0, 10]", lambda size: rng.uniform(0, 10, size)),
                   ("Normal, mean 5, sd 2", lambda size: rng.normal(5, 2, size))]
    titles = [f"{name} — {part}" for name, _ in populations
              for part in ("population", "means of n = 5", "means of n = 30")]
    fig = make_subplots(rows=3, cols=3, subplot_titles=titles, vertical_spacing=0.09)
    for row, (name, draw) in enumerate(populations, start=1):
        fig.add_histogram(x=draw(10_000), nbinsx=50, marker_color=GREY, showlegend=False, row=row, col=1)
        for col, size in ((2, 5), (3, 30)):
            fig.add_histogram(x=draw((1000, size)).mean(axis=1), nbinsx=30, marker_color=BLUE,
                              showlegend=False, row=row, col=col)
    fig.update_annotations(font_size=11)
    fig.update_layout(title="The central limit theorem, redrawn (illustrative)")
    show(fig, "clt_grid", height=820)
    ''', slides=["19"], treatment="illustrative: the slide's simulation redone with numpy; "
                                 "parameters read off its axes")
    explain(
        "Test the theorem's first condition on the slice.",
        "The definition slide says this archive breaks independence: consecutive readings "
        "correlate at 0.9967, so n must be replaced by the effective sample size "
        "[@bayley1946]. Averaging consecutive readings is therefore not the averaging the "
        "theorem describes.",
        "Takes the payload readings of 22 January (skewed). Draws 2,000 means of 30 readings "
        "chosen independently at random, and 2,000 means of 30 *consecutive* readings "
        "(15 seconds), and compares their skewness. Then computes the slice's effective "
        "sample size n(1 − ρ)/(1 + ρ).",
        "Independent draws give a nearly symmetric mean; consecutive readings give a mean "
        "almost as skewed as the readings themselves. The theorem applies to window features "
        "only when the windows, not the readings, are the observations.")
    nb.figure("clt_on_slice", r'''
    payload = slice_day_one["payload"].to_numpy(float)
    rng = np.random.default_rng(20200122)
    independent = rng.choice(payload, size=(2000, 30), replace=True).mean(axis=1)
    starts = rng.integers(0, len(payload) - 30, 2000)
    consecutive = np.array([payload[s:s + 30].mean() for s in starts])
    panels = [("the readings, 22 January", payload, GREY),
              ("means of 30 independent draws", independent, BLUE),
              ("means of 30 consecutive readings", consecutive, ORANGE)]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[p[0] for p in panels])
    for col, (label, values, colour) in enumerate(panels, start=1):
        fig.add_histogram(x=values, nbinsx=40, marker_color=colour, showlegend=False, row=1, col=col)
        fig.update_xaxes(title_text=f"kg — skewness {stats.skew(values):.2f}", row=1, col=col)
    fig.update_layout(title="The theorem needs independence: the same averaging, two kinds of sample")
    show(fig, "clt_on_slice", height=420)
    for label, values, _ in panels:
        print(f"  {label:<36} skewness {stats.skew(values):5.2f}")
    rho = lag_one(in_time["speed"])
    agrees("effective sample size of the slice, n(1 − ρ)/(1 + ρ)",
           len(in_time) * (1 - rho) / (1 + rho), module1("slice_effective_sample_size"), 0)
    ''', slides=["19"], treatment="lab data: the slide's theorem tested on the slice's payload, "
                                 "independent draws against consecutive readings")

    # --- slide 20: the normal distribution -------------------------------------------
    nb.md("### Properties of the normal distribution\n\n"
          "*Slide: \"Properties of normal distribution\".*")
    explain(
        "Write the two formulas on the slide: the normal density and the z-score.",
        "The 68-95-99.7 rule is a property of this one curve; it is not a property of data.",
        "Writes both in sympy, checks that the density integrates to one, and computes the "
        "probability within one, two and three standard deviations exactly, as "
        "erf(k/√2).",
        "68.27, 95.45 and 99.73 per cent — the rule's three numbers, derived rather than "
        "remembered.")
    nb.equation("normal_density", r'''
    x = sp.Symbol("x", real=True)
    mu = sp.Symbol(r"\mu", real=True)
    sigma = sp.Symbol(r"\sigma", positive=True)
    density = sp.exp(-(x - mu)**2 / (2 * sigma**2)) / sp.sqrt(2 * sp.pi * sigma**2)
    formula(sp.Eq(sp.Function("f")(x), density, evaluate=False))
    print("integral over the real line:", sp.simplify(sp.integrate(density, (x, -sp.oo, sp.oo))))
    within = {k: sp.erf(k / sp.sqrt(2)) for k in (1, 2, 3)}
    for k, stated in zip(within, (68, 95, 99.7)):
        agrees(f"per cent within {k} standard deviation(s)", 100 * float(within[k]), stated, 0 if k < 3 else 1)
    ''', slides=["20"])
    explain(
        "Write the z-score the slide pairs with the density.",
        "It is the distance from the mean in standard deviations — the quantity the "
        "z-score outlier rule of Block 1 thresholds, and a meaningful one only when the "
        "column is close to this curve.",
        "Writes Z = (X − μ)/σ in sympy.",
        "Under the normal curve |Z| > 3 happens 0.27 per cent of the time; the outlier "
        "counts above give 0.67 per cent on the slice's payload and 2.6 per cent on its "
        "speed — the rule's promise holds only for the shape it assumes.")
    nb.equation("z_score", r'''
    X = sp.Symbol("X", real=True)
    formula(sp.Eq(sp.Symbol("Z"), (X - mu) / sigma, evaluate=False))
    ''', slides=["20"])
    explain(
        "Draw the normal curve with its three bands.",
        "The slide's picture is a generated drawing whose formula is garbled and whose "
        "axis is labelled μ − 3σ, μ − σ, μ, μ + σ, μ + 2σ — not symmetric. Redrawn from the "
        "closed form, it cannot be.",
        "Evaluates the standard normal density and shades ±1, ±2 and ±3 standard deviations, "
        "with the shares computed above.",
        "Symmetric, mean = median = mode, and tails that never reach the axis — each "
        "property visible, none asserted.")
    nb.figure("normal_properties", r'''
    grid = np.linspace(-4, 4, 801)
    fig = go.Figure()
    for k, shade in ((3, "rgba(42,120,214,0.12)"), (2, "rgba(42,120,214,0.22)"), (1, "rgba(42,120,214,0.38)")):
        inside = np.abs(grid) <= k
        fig.add_scatter(x=grid[inside], y=stats.norm.pdf(grid[inside]), fill="tozeroy", mode="none",
                        fillcolor=shade, name=f"within ±{k}σ: {100 * float(within[k]):.2f}%")
    fig.add_scatter(x=grid, y=stats.norm.pdf(grid), mode="lines", line=dict(color=NAVY, width=3),
                    name="f(x), μ = 0, σ = 1")
    fig.add_vline(x=0, line=dict(color=GREY, dash="dash"), annotation_text="mean = median = mode")
    fig.update_xaxes(tickvals=[-3, -2, -1, 0, 1, 2, 3],
                     ticktext=["μ − 3σ", "μ − 2σ", "μ − σ", "μ", "μ + σ", "μ + 2σ", "μ + 3σ"])
    fig.update_layout(title="The normal distribution and the 68-95-99.7 rule",
                      yaxis_title="density", legend=dict(x=0.72, y=0.95))
    show(fig, "normal_properties", height=460)
    beside("axis labels on the slide's drawing", "μ − 3σ, μ − 2σ, μ − σ, μ, μ + σ, μ + 2σ, μ + 3σ",
           "μ − 3σ, μ − σ, μ, μ + σ, μ + 2σ",
           "the generated drawing omits μ − 2σ and μ + 3σ, and its density formula lacks the σ "
           "under the root and has a garbled exponent; the slide's own formula image is correct")
    ''', slides=["20"], treatment="exact: closed form (the slide's generated drawing redrawn)")

    # --- slide 21: six faults -----------------------------------------------------------
    nb.md("### Six faults, measured before any is repaired\n\n"
          "*Slide: \"Six faults occur in most files, and types must be repaired first\".*")
    explain(
        "Measure the six faults the slide lists, on the slice and the generated phones.",
        "Measure all six before repairing any, and repair types first, because every other "
        "measurement depends on them: a time read as text cannot be floored to a window, "
        "and a number read as text has no range. (A rule of practice: the slide gives no "
        "source for it, and none is cited here.)",
        "Counts: missing values per column; duplicated rows and duplicated instants; the "
        "distinct spellings of text columns; the types pandas inferred (a time column read "
        "as text); outliers (the broken counter, negative speeds); and structural errors — "
        "columns that do not hold what their names claim.",
        "Nothing here is repaired. The table is the audit; every later decision can point "
        "at a line of it.")
    nb.code(r'''
    missing = {c: round(float(bus[c].isna().mean()) * 100, 2) for c in bus.columns if bus[c].isna().any()}
    print("1 missing, per cent of rows:", missing)
    print("2 duplicated rows:", int(bus.duplicated().sum()), "  duplicated (vehicle, instant):",
          int(bus.duplicated(["vehicle_id", "utc_time"]).sum()))
    print("3 spellings: door_state", sorted(bus["door_state"].dropna().unique()),
          "  phone label", sorted(phones["label"].unique()))
    print("4 types as read: utc_time", bus["utc_time"].dtype, "  timestamp", bus["timestamp"].dtype,
          "  mileage", bus["mileage"].dtype)
    print("5 outliers: mileage at 65,535:", int((bus["mileage"] == 65535).sum()),
          "  speed below zero:", int((bus["speed"] < 0).sum()))
    offset = (pd.to_datetime(bus["timestamp"]) - pd.to_datetime(bus["utc_time"])).dt.total_seconds() / 3600
    constant = [c for c in bus.columns if bus[c].nunique(dropna=True) == 1]
    print("6 structural: `timestamp` minus `utc_time`, hours:", [float(v) for v in sorted(offset.round(2).unique())],
          "  `emergency_stop` holds", sorted(bus["emergency_stop"].dropna().unique()),
          "  constant columns:", constant)
    beside("rows with negative speed", int((bus["speed"] < 0).sum()),
           f"{module1('negative_speed_rows'):,} (archive, Module 1)",
           "Module 1 counted both vehicles in the archive's bus file; the slice holds one")
    ''')

    # --- slides 22-26: the grain, the ledger, the profile -------------------------------
    nb.md("""
    ### The grain, and the ledger

    *Slides: "The two sources share no timestamps, so a grain must be chosen" (its
    figure, the two clocks, is drawn in the introduction), "Definition — a tumbling
    window on the coordinated universal time grain", "Definition — the conservation
    ledger" and "Definition — the upstream profile, and checking a frame against it".*
    """)
    explain(
        "Write the tumbling window, and check that flooring a timestamp computes it.",
        "A tumbling window is one of a sequence of fixed-length intervals, closed on the "
        "left and open on the right, so every reading belongs to exactly one "
        "[@akidau2015, § 1.2].",
        "Writes w_j as a sympy interval, closed on the left and open on the right, and the "
        "index j = ⌊(t − t₀)/Δ⌋; checks symbolically that a reading at t lies in w_j for "
        "that j; then checks, on every vehicle reading, that pandas' `dt.floor(\"5s\")` — "
        "what Lab 1's `align` uses — gives t₀ + jΔ.",
        "The formula on the slide and the line of code in the solution are the same "
        "function, reading for reading.")
    nb.equation("tumbling_window", r'''
    t, t0 = sp.symbols("t t_0", real=True)
    Delta = sp.Symbol(r"\Delta", positive=True)
    j = sp.Symbol("j", integer=True)
    window_j = sp.Interval(t0 + j * Delta, t0 + (j + 1) * Delta, right_open=True)
    index_of_t = sp.floor((t - t0) / Delta)
    formula(sp.Eq(sp.Symbol("w_j"), window_j, evaluate=False), sp.Eq(j, index_of_t, evaluate=False),
            sp.Eq(Delta, sp.Symbol(r"5\ \mathrm{s}"), evaluate=False))
    u = sp.Symbol("u", real=True)       # u = (t − t0)/Δ, so t = t0 + uΔ
    inside = window_j.subs(j, sp.floor(u)).contains(t0 + u * Delta)
    print("t0 + u·Δ lies in w_j with j = floor(u) (t0 = 0, Δ = 5, at 1,000 values of u):",
          all(bool(inside.subs({u: sp.Rational(k, 7), t0: 0, Delta: 5})) for k in range(-500, 500)))
    stamps = pd.to_datetime(bus["utc_time"], utc=True)
    midnight = stamps.dt.normalize()
    index = np.floor((stamps - midnight).dt.total_seconds() / 5).astype(int)
    by_formula = midnight + pd.to_timedelta(index * 5, unit="s")
    print("readings where floor('5s') differs from t0 + j·Δ:", int((stamps.dt.floor("5s") != by_formula).sum()),
          "of", f"{len(stamps):,}")
    ''', slides=["23"])
    explain(
        "Write the ledger's identity.",
        "Every input row is used or dropped, never both, and every dropped row has a reason "
        "with a count. The identity holds by construction where counts are kept; it is "
        "bookkeeping, and needs no source. (The Lab 1 stub cites Wang and Strong (1996) "
        "for it; that paper defines data quality as fitness for use and does not state "
        "this identity.)",
        "Writes the two equations in sympy, with r running over the K reasons, and "
        "evaluates both on a planted ledger of three reasons.",
        "Lab 1's check adds the buckets up; a ledger that does not balance is the lesson.")
    nb.equation("ledger", r'''
    used, dropped, received = sp.symbols(r"\mathrm{used} \mathrm{dropped} \mathrm{received}",
                                         nonnegative=True, integer=True)
    r, K = sp.symbols("r K", positive=True, integer=True)
    drop_reasons = sp.IndexedBase(r"\mathrm{drop\_reasons}")
    formula(sp.Eq(used + dropped, received, evaluate=False),
            sp.Eq(sp.Sum(drop_reasons[r], (r, 1, K)), dropped, evaluate=False))
    planted = {drop_reasons[1]: 6, drop_reasons[2]: 40, drop_reasons[3]: 0}
    dropped_here = sp.Sum(drop_reasons[r], (r, 1, 3)).doit().subs(planted)
    print("planted: 10,754 used, reasons 6 + 40 + 0 → dropped =", dropped_here,
          "; balances against 10,800 received:", sp.Eq(10_754 + dropped_here, 10_800))
    ''', slides=["25"])
    explain(
        "Write what \"checking a frame against the upstream profile\" means, and run it.",
        "Module 1 declared the slice's columns — type, unit, range, tolerated absence, "
        "sampling step — and `check_against` runs that declaration. The order of the rules "
        "is part of the design: type before range. \"Fit for use\" is Wang and Strong's "
        "definition of data quality: judged by the consumer of the data, here the next "
        "step of the pipeline [@wang1996].",
        "Writes the definition in sympy — the number of breaches as the sum of the "
        "complaints of the five rules, ρ = 1 … 5 (presence, type, range, absence, step, in "
        "that order), and fitness for use as that number being nought — then runs "
        "`check_against` on the whole slice and on 22 January alone.",
        "The pooled frame satisfies the declaration; its first day does not. An examination "
        "made only on the pool would report nothing — the shape of the paradox that ends "
        "this block.")
    nb.equation("profile_breaches", r'''
    X, P = sp.symbols("X P")
    rho = sp.Symbol(r"\rho", positive=True, integer=True)
    breaches = sp.Abs(sp.Function(r"\mathrm{breaches}")(X, P))
    complaints = sp.Function(r"n_{\mathrm{complaints}}")
    formula(sp.Eq(breaches, sp.Sum(complaints(rho, X, P), (rho, 1, 5)), evaluate=False),
            sp.Equivalent(sp.Symbol(r"X\ \text{fit for use}"), sp.Eq(breaches, 0), evaluate=False))
    profile = load_module1_profile()
    print("both days:", check_against(bus, profile) or "no complaints")
    day_22 = bus[pd.to_datetime(bus["utc_time"], utc=True).dt.date.astype(str) == "2020-01-22"]
    print("22 January alone:", check_against(day_22, profile))
    ''', slides=["26"])

    # --- slides 27-28: Simpson ---------------------------------------------------------
    nb.md("### Simpson's paradox\n\n*Slides: \"Definition — Simpson's paradox\" and \"The aboard "
          "share rises on each shuttle and falls across the pooled days\".* The definition "
          "slide's cartoon is not redrawn (copyrighted characters); the paradox is drawn from "
          "the planted data instead.")
    explain(
        "Write the reversal as the slide does, and verify it on the generator's "
        "two-by-two-by-two table.",
        "An association can point one way inside every group and the other way pooled, "
        "because pooling also compares the sizes of the groups [@simpson1951; @blyth1972]. "
        "Which number to report depends on what changed the mix [@pearl2014]; road-safety "
        "examples are in {@elvik2025}.",
        "Writes the slide's two inequalities as sympy relations, P(Y | D = d, G = g) being a "
        "symbol for each share; then reads the counts from `simpson_table()` — the "
        "generator's own function — substitutes them as exact fractions, with D = 1 the "
        "first day (22 January) and D = 0 the second, and lets sympy decide each relation.",
        "Every inequality holds exactly as the slide writes it: the pooled share is higher "
        "on the first day, and each shuttle's share is lower.")
    nb.equation("simpson", r'''
    g = sp.Symbol("g")
    def share_symbol(day, group=None):
        return sp.Symbol(rf"P(Y \mid D{{=}}{day}" + (rf", G{{=}}{group})" if group is not None else ")"))
    pooled_rises = sp.StrictGreaterThan(share_symbol(1), share_symbol(0), evaluate=False)
    each_falls = sp.StrictLessThan(share_symbol(1, g), share_symbol(0, g), evaluate=False)
    formula(pooled_rises, sp.Symbol(r"\text{while}"), each_falls, sp.Symbol(r"\text{for every } g"))
    counts = simpson_table()
    share = {(row.group, row.day): sp.Rational(int(row.aboard), int(row.readings))
             for row in counts.itertuples()}
    d1, d0 = "2020-01-22", "2020-01-23"
    print("pooled:", share[("pooled", d1)], ">", share[("pooled", d0)], "is",
          pooled_rises.subs({share_symbol(1): share[("pooled", d1)], share_symbol(0): share[("pooled", d0)]}))
    for group in ("Bus 1", "Bus 2"):
        print(f"{group}:", share[(group, d1)], "<", share[(group, d0)], "is",
              each_falls.subs({share_symbol(1, g): share[(group, d1)], share_symbol(0, g): share[(group, d0)]}))
    ''', slides=["27"])
    explain(
        "Draw the reversal with the deck's own figure code.",
        "The slide's figure comes from `slides/make_figs.py`; running its function verbatim "
        "is the surest way to reproduce it.",
        "Copies the palette and the constants from `make_figs.py`; `write()` is the one "
        "change — there it writes `figures/<name>.png`, here it applies the same layout and "
        "shows the figure. Then copies `figure_simpson()`, which imports `simpson_table` "
        "from `make_phones` — the generator defined above.",
        "Bus 1 from 40.1 to 45.0 per cent, Bus 2 from 64.4 to 70.0, pooled from 56.3 to "
        "51.2: the slide's numbers, recomputed.")
    nb.source(FIGS, "BLUE", "ORANGE", "GREY", "RED", "TEMPLATE", "BEACON", "DAY_ONE", "DAY_TWO")
    explain(
        "Replace `make_figs.py`'s `write()` with one that shows the figure.",
        "The script writes PNG and HTML files under `slides/figures/`; the notebook must "
        "not write there.",
        "Applies the script's own layout (template, size, margins) and displays the image.",
        "The figures below are the deck's, drawn by the deck's code, in place.")
    nb.code(r'''
    def write(fig, name, width=1000, height=560):
        """make_figs.py's write(), shown in place instead of written to figures/."""
        fig.update_layout(template=TEMPLATE, width=width, height=height,
                          margin=dict(l=70, r=30, t=70, b=70))
        display(Image(fig.to_image(format="png", width=width, height=height, scale=1)))
    ''')
    explain(
        "Copy the figure function.",
        "It is the code that drew the slide.",
        "`figure_simpson()` from `make_figs.py`, verbatim: it measures the table from the "
        "generator, draws it, and returns the table and the movements in points.",
        "The next cell calls it.")
    nb.source(FIGS, "figure_simpson")
    explain(
        "Draw the slide's figure and check its six shares and three movements.",
        "The slide quotes 40.1 → 45, 64.4 → 70, 56.3 → 51.2, and −5.1 against +4.9 and "
        "+5.6 points.",
        "Calls `figure_simpson()` and compares every number with the slide.",
        "All nine agree.")
    nb.figure("simpson", r'''
    table, points = figure_simpson()
    for group, before, after in (("Bus 1", 40.1, 45), ("Bus 2", 64.4, 70), ("pooled", 56.3, 51.2)):
        agrees(f"{group}, 22 January, per cent aboard", table[group]["2020-01-22"]["share"], before, 1)
        agrees(f"{group}, 23 January, per cent aboard", table[group]["2020-01-23"]["share"], after, 1)
    for group, stated in (("pooled", -5.1), ("Bus 1", 4.9), ("Bus 2", 5.6)):
        agrees(f"{group}, change in points", points[group], stated, 1)
    ''', slides=["28"], treatment="exact: the slide's own code")

    # --- Lab 1 -------------------------------------------------------------------------
    nb.md("""
    ## Laboratory 1 — Audit and align

    *Slide: "Lab 1 — Audit and align".* Two functions, twenty-five minutes. The check
    adds the ledger's buckets up, confirms which clock was used, runs Module 1's profile
    on three frames whose answers differ, and grades the reversal and its name.
    """)
    nb.statement(f"{LABS}/01_audit_and_align.py")
    nb.md("""
    **The stub's slide titles, against the deck.** The stub places the lab under "Bronze
    to silver — what changes, and what must not"; no slide has that title. Block one opens
    with "Know the distribution, then clean", and the bronze-and-silver rule is the slide
    "Silver is derived and rebuildable, and bronze must never be modified", at the start
    of Block two. The four definition slides it names carry exactly those titles. The
    stub also cites Wang and Strong (1996) twice: for the profile ("fit for use" is their
    definition of data quality) it fits; for the conservation ledger it does not — the
    paper states no such identity, which is bookkeeping. The closing section checks every
    title the stubs quote against the deck.
    """)
    nb.md("### The solution")
    explain(
        "Define the two functions as the Lab 1 solution writes them.",
        "`align` makes the three decisions explicitly — which clock, which grain "
        "[@akidau2015], what happens to the leftovers — keeps the books, and records Module "
        "1's verdict on the frame; `pooled_versus_by_group` reports both comparisons and "
        "names the reversal [@simpson1951; @blyth1972; @pearl2014].",
        "Copies `LAB`, `SIMPSON`, `align` and `pooled_versus_by_group` from "
        "`solutions/lab_01.py`, with their docstrings.",
        "Two functions; the next cells run them as the stub and the solution do.")
    nb.source(f"{S}/lab_01.py", "LAB", "SIMPSON", "align", "pooled_versus_by_group",
              cite="[@akidau2015; @simpson1951; @blyth1972; @pearl2014]")
    explain(
        "Register the solved Lab 1 under the name the other labs import it by.",
        "The Lab 4 solution does `from lab_01 import align`.",
        "Puts a module named `lab_01` holding the two functions into `sys.modules`.",
        "The import in Lab 4 will find the function above, not the file.")
    nb.code(r'''
    as_module("lab_01", align=align, pooled_versus_by_group=pooled_versus_by_group, SIMPSON=SIMPSON)
    ''')
    explain(
        "Run the stub's own demonstration against the solved functions.",
        "It is what a student sees when the file is complete.",
        "The lines below are the stub's `__main__` block, verbatim.",
        "An aligned table, the ledger, Module 1's verdict on the vehicle frame, and the "
        "reversal.")
    nb.step(f"{LABS}/01_audit_and_align.py", 0)
    explain(
        "Ask the ledger the archive's question about unparseable timestamps.",
        "The archive's phone file loses 6 rows when a reader infers one timestamp format, "
        "and none when told the formats are mixed (the slide saying so is hidden; the "
        "ledger's slide names the reason). The generated phones are clean, so the ledger's "
        "reason \"timestamp would not parse\" counts 0 here.",
        "Runs `align` on the generated day, then on a copy with six timestamps made "
        "unreadable, and prints the archive's recorded counts beside.",
        "The reason and its count appear in the ledger either way; that is what makes "
        "\"six unparseable rows\" a recorded decision rather than a silent loss.")
    nb.code(r'''
    _, ledger = align(bus, phones, 5)
    broken = phones.copy()
    broken["timestamp_utc"] = broken["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
    broken.loc[broken.index[-6:], "timestamp_utc"] = "not a time"
    _, broken_ledger = align(bus, broken, 5)
    print("generated day:", ledger["drop_reasons"])
    print("six timestamps broken:", broken_ledger["drop_reasons"],
          "— balances:", broken_ledger["phone_rows_used"] + broken_ledger["phone_rows_dropped"]
          == broken_ledger["phone_rows_in"])
    beside("unparseable phone timestamps", ledger["drop_reasons"]["timestamp would not parse"],
           f"{archive('unparseable_timestamps')} when told the formats are mixed, "
           f"{archive('unparseable_if_one_format_assumed')} when one format is inferred (archive)",
           "the generated timestamps are all well formed; the archive's six are a property "
           "of how its file is read")
    ''')
    nb.md("""
    ### The solution's demonstration, step by step

    What `python3 solutions/lab_01.py` prints, run here one paragraph of its `__main__`
    block at a time, each verbatim. It adds what the stub's demonstration does not: an
    interrupted vehicle day, where the ledger has something to account for; the profile
    run on one day against both; and the reversal's figure.
    """)
    lab1 = f"{S}/lab_01.py"
    explain(
        "Start the demonstration's narrator.",
        "Every line the solution prints goes through one logger, so that the terminal and "
        "this notebook show the same story in the same words.",
        "Creates `say`, the narrator for Lab 1, and prints the lab's one-line summary.",
        "Each line below begins with the seconds since this cell ran — the one part of the "
        "output that changes from run to run.")
    nb.paragraph(lab1, 1)
    explain(
        "Align the shipped pair on the five-second grid and read the ledger.",
        "This is the lab's first deliverable at work on the real slice and the first "
        "generated day: one clock (`utc_time`), one grain (5 s, because neither 0.5 s nor "
        "0.993 s divides the other), and the books kept.",
        "Loads both files, calls `align(bus, phones, 5)`, prints the table's size, Module "
        "1's verdict on the vehicle frame, the ledger as a table, the drop reasons, and the "
        "conservation identity used + dropped = received.",
        "On the shipped pair nothing is dropped and the identity holds trivially; the next "
        "paragraph gives the ledger something to count.")
    nb.paragraph(lab1, 2)
    explain(
        "Cut the vehicle day in half and align again.",
        "A ledger that only ever balances on complete data proves nothing; this is the day "
        "the check also uses.",
        "Keeps the vehicle readings before the midpoint of the phones' span, aligns, and "
        "prints how many phone rows were dropped and why.",
        "Every phone row after the vehicle stopped is dropped with a reason, and used + "
        "dropped still equals received.")
    nb.paragraph(lab1, 3)
    explain(
        "Run Module 1's profile on one day of the vehicle frame instead of both.",
        "The profile is an examination; an examination made only on the pooled frame can "
        "miss what one day breaks.",
        "Keeps 22 January's vehicle readings, aligns them, and prints the profile's verdict "
        "on both days beside its verdict on the one day.",
        "No complaints on the pool, complaints on 22 January alone: the module's paradox in "
        "a validation layer.")
    nb.paragraph(lab1, 4)
    explain(
        "Compare the two generated days pooled and per shuttle.",
        "This is the second deliverable: `pooled_versus_by_group` must report both "
        "comparisons and name the reversal when every group moves against the pool "
        "[@simpson1951; @blyth1972].",
        "Loads the two-day Simpson frame, calls `pooled_versus_by_group(simpson, \"aboard\", "
        "\"bus\", \"day\")`, and prints the shares in per cent, the pooled difference, each "
        "shuttle's difference, and the verdict.",
        "The pooled share falls from the first day to the second while each shuttle's "
        "rises; the function says so, and names it.")
    nb.paragraph(lab1, 5)
    explain(
        "Draw the reversal.",
        "The slide's figure, drawn by the lab from the verdict it just returned.",
        "Grouped bars of the aboard share, per shuttle and pooled, one colour per day; "
        "`save_figure` shows it here.",
        "Up on each shuttle, down when pooled, in one picture.")
    nb.paragraph(lab1, 6)
    explain(
        "Print what the check grades.",
        "The demonstration ends by telling the student what `verify/check_01.py` will ask.",
        "One narrated line.",
        "The ledger is graded on two days, the grid on its grain, and the reversal on a "
        "planted frame, not on the shipped one.")
    nb.paragraph(lab1, 7)
