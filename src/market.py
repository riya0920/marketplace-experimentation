"""An agent-based delivery marketplace, built to make interference REAL.

The whole project depends on this simulator having an honest interference
channel, so it is worth stating the causal chain explicitly:

    treatment raises conversion
        -> more orders are placed
            -> more couriers are busy
                -> utilisation rises
                    -> quoted delivery time rises
                        -> conversion falls FOR EVERYONE in that region,
                           including the control group

That last step is the SUTVA violation. Control users are not unaffected
bystanders; they are competing for the same couriers, and the treatment makes
them worse off. A naive A/B test measures (treated - control) and therefore
counts the treatment's benefit AND the harm it did to its own comparison group,
which is why it overstates.

Nothing here is calibrated to a real company. It is a LAB: the point is that the
truth is knowable by construction, so estimators can be scored. Stylised facts it
is checked against (see validate()):
  - delivery time rises with utilisation, steeply as utilisation -> 1
  - demand is time-varying with lunch and dinner peaks
  - conversion falls as quoted delivery time rises
"""
from __future__ import annotations

import numpy as np

MINUTES_PER_DAY = 1440


def demand_curve(minute_of_day: np.ndarray) -> np.ndarray:
    """Two peaks: lunch around 12:30, dinner around 18:30."""
    h = minute_of_day / 60.0
    lunch = np.exp(-0.5 * ((h - 12.5) / 1.1) ** 2)
    dinner = np.exp(-0.5 * ((h - 18.5) / 1.6) ** 2)
    base = 0.12 + 0.05 * np.exp(-0.5 * ((h - 9.0) / 2.5) ** 2)
    return base + 1.0 * lunch + 1.35 * dinner


# Monday..Sunday. Friday and Saturday are the ones that matter, and a simulator
# where every day is statistically identical cannot represent the single most
# common real-world confound in a short experiment: the test that ran Tuesday to
# Thursday and the one that caught a weekend are not the same test.
DOW_MULTIPLIER = np.array([0.88, 0.86, 0.92, 1.06, 1.34, 1.42, 1.12])

# Daily trend: the marketplace is growing. A five-day A/B with a trend and an
# unbalanced assignment across days will attribute the trend to the treatment,
# and no amount of within-day randomisation fixes it.
DAILY_TREND = 0.004


def day_multiplier(day: int, rng: np.random.Generator | None = None,
                   weather_sd: float = 0.12) -> float:
    """Day-of-week x trend x weather.

    Weather is a per-DAY shock shared across every region, which is what makes it
    a genuine confound rather than noise: a rainy Tuesday raises demand
    everywhere at once, so it cannot be averaged away across regions the way
    region-specific noise can.
    """
    dow = DOW_MULTIPLIER[day % 7]
    trend = (1.0 + DAILY_TREND) ** day
    shock = 1.0
    if rng is not None and weather_sd > 0:
        shock = float(np.exp(rng.normal(0.0, weather_sd)))
    return dow * trend * shock


class Marketplace:
    """Minute-stepped, region-partitioned, vectorised across regions.

    Parameters
    ----------
    n_regions      : geographic clusters -- the unit a geo experiment randomises
    couriers       : couriers per region. LOWER = tighter market = more interference
    arrival_scale  : customers arriving per region-minute at peak
    base_eta       : minutes of delivery time at zero utilisation
    eta_slope      : how hard congestion bites (the 1/(1-u) term's coefficient)
    beta_eta       : conversion sensitivity to quoted ETA, per minute
    """

    def __init__(self, n_regions=12, couriers=14, arrival_scale=0.40,
                 base_eta=18.0, eta_slope=13.0, beta_eta=0.020,
                 base_conversion=0.62, service_minutes=34.0, seed=0,
                 weather_seed=20260101):
        self.R = n_regions
        self.couriers = np.full(n_regions, couriers, dtype=float)
        self.arrival_scale = arrival_scale
        self.base_eta = base_eta
        self.eta_slope = eta_slope
        self.beta_eta = beta_eta
        self.base_conversion = base_conversion
        self.service_minutes = service_minutes
        self.rng = np.random.default_rng(seed)
        # deliberately NOT derived from `seed`: every design being compared must
        # draw the same weather, or a design comparison is partly a comparison
        # of the weather each design happened to get.
        self.weather_seed = int(weather_seed)
        # heterogeneous regions: a geo experiment has to survive this
        self.region_mult = 0.65 + 0.7 * np.random.default_rng(seed + 991).random(n_regions)

    # ------------------------------------------------------------------
    def eta(self, busy: np.ndarray) -> np.ndarray:
        """Quoted delivery time as a function of utilisation.

        The 1/(1-u) shape is the queueing result, capped so a saturated region
        quotes something finite rather than blowing up the simulation.
        """
        u = np.clip(busy / self.couriers, 0.0, 0.985)
        return self.base_eta + self.eta_slope * (u / (1.0 - u))

    def conversion(self, eta_minutes: np.ndarray, treated: np.ndarray,
                   lift: float) -> np.ndarray:
        """P(order | saw this ETA). Treatment multiplies the odds-free rate."""
        p = self.base_conversion * np.exp(-self.beta_eta * (eta_minutes - self.base_eta))
        p = p * np.where(treated, 1.0 + lift, 1.0)
        return np.clip(p, 0.0, 0.999)

    # ------------------------------------------------------------------
    def run(self, days: int, assign_fn, lift: float, warmup_minutes: int = 240,
            pool=None, reposition_every: int = 30, targeted: bool = False,
            compliance: float = 0.55, shrink: float = 1.0):
        """Simulate. `assign_fn(minute, region_idx, n_customers, rng) -> bool array`
        decides which of this minute's customers are treated.

        `pool` is an optional `mobility.CourierPool`. With one, idle couriers
        move between regions every `reposition_every` minutes under SE-3's
        policy, and per-region capacity stops being constant. That is the
        difference between a simulator in which a geo design's independence
        assumption holds by construction and one in which it can be tested.

        Returns per-(minute, region) records of exposures, orders, treated
        exposures/orders, utilisation and ETA.
        """
        T = days * MINUTES_PER_DAY
        busy_until: list[list[float]] = [[] for _ in range(self.R)]
        busy = np.zeros(self.R)
        # One weather draw per DAY, shared across regions. Drawn up front so the
        # same seed gives the same weather to every design being compared --
        # otherwise a design comparison is partly a comparison of the weather it
        # happened to get.
        # NOT hash(): Python salts str/tuple hashing per process, so
        # `hash(("weather", days))` gave a DIFFERENT weather sequence on every
        # run of the interpreter. The comment above -- "the same seed gives the
        # same weather to every design being compared" -- was true within one
        # process and false across two, which is the worst of both: comparisons
        # inside a report were sound, and nobody could reproduce the report.
        # Caught when the same sweep printed different t-statistics on a second
        # run with every seed unchanged.
        wrng = np.random.default_rng(self.weather_seed + days)
        day_mult = np.array([day_multiplier(d, wrng) for d in range(days)])

        base_couriers = self.couriers.copy()
        if pool is not None:
            self.couriers = np.maximum(pool.counts(), 1.0)
        # a trailing estimate of arrivals per region, which is what a dispatch
        # system would actually have -- it cannot see the Poisson rate
        seen = np.zeros(self.R)

        n_rec = T * self.R
        rec = dict(
            minute=np.zeros(n_rec, dtype=np.int32),
            region=np.zeros(n_rec, dtype=np.int16),
            exposures=np.zeros(n_rec, dtype=np.int32),
            orders=np.zeros(n_rec, dtype=np.int32),
            t_exposures=np.zeros(n_rec, dtype=np.int32),
            t_orders=np.zeros(n_rec, dtype=np.int32),
            c_exposures=np.zeros(n_rec, dtype=np.int32),
            c_orders=np.zeros(n_rec, dtype=np.int32),
            utilisation=np.zeros(n_rec), eta=np.zeros(n_rec))
        k = 0

        for t in range(T):
            # couriers finishing this minute become free
            for r in range(self.R):
                if busy_until[r]:
                    keep = [x for x in busy_until[r] if x > t]
                    busy_until[r] = keep
                    busy[r] = len(keep)

            mod = t % MINUTES_PER_DAY
            lam = (self.arrival_scale * demand_curve(np.array([mod]))[0]
                   * self.region_mult * day_mult[t // MINUTES_PER_DAY])
            arrivals = self.rng.poisson(lam)
            seen = 0.97 * seen + 0.03 * arrivals
            if (pool is not None and t % reposition_every == 0
                    and t >= warmup_minutes):
                util = np.clip(busy / np.maximum(self.couriers, 1.0), 0, 1)
                surge = 1.0 + 1.2 * np.clip(util - 0.55, 0, None)
                self.couriers = np.maximum(
                    pool.step(busy, seen * reposition_every, surge=surge,
                              targeted=targeted, compliance=compliance,
                              shrink=shrink), 1.0)
            cur_eta = self.eta(busy)

            for r in range(self.R):
                n = int(arrivals[r])
                idx = k + r
                rec["minute"][idx] = t
                rec["region"][idx] = r
                rec["utilisation"][idx] = busy[r] / self.couriers[r]
                rec["eta"][idx] = cur_eta[r]
                if n == 0:
                    continue
                treated = assign_fn(t, r, n, self.rng)
                p = self.conversion(np.full(n, cur_eta[r]), treated, lift)
                converted = self.rng.random(n) < p

                rec["exposures"][idx] = n
                rec["orders"][idx] = int(converted.sum())
                rec["t_exposures"][idx] = int(treated.sum())
                rec["t_orders"][idx] = int((converted & treated).sum())
                rec["c_exposures"][idx] = int((~treated).sum())
                rec["c_orders"][idx] = int((converted & ~treated).sum())

                # each order occupies a courier; excess demand simply queues by
                # raising utilisation on subsequent minutes
                for _ in range(int(converted.sum())):
                    dur = max(5.0, self.rng.normal(self.service_minutes, 8.0))
                    busy_until[r].append(t + dur)
                busy[r] = len(busy_until[r])
            k += self.R

        for key in rec:
            rec[key] = rec[key][:k]
        rec["warmup"] = rec["minute"] < warmup_minutes
        if pool is not None:
            rec["final_couriers"] = self.couriers.copy()
            rec["courier_moves"] = pool.moves
            self.couriers = base_couriers
        return rec


# ----------------------------------------------------------------------
# assignment functions -- the experiment designs
# ----------------------------------------------------------------------
def assign_all(flag: bool):
    def f(t, r, n, rng):
        return np.full(n, flag)
    return f


def assign_user_level(p: float = 0.5):
    """Naive A/B: each arriving customer is independently randomised."""
    def f(t, r, n, rng):
        return rng.random(n) < p
    return f


def assign_switchback(block_minutes: int, seed: int = 0):
    """Time-sliced: the WHOLE market is treated or control for a block.

    Randomising the market rather than the user is what removes the
    interference: within a block there is no control group being harmed, because
    there is no control group at all.
    """
    rng = np.random.default_rng(seed)
    draws: dict[int, bool] = {}

    def f(t, r, n, rng_unused):
        b = t // block_minutes
        if b not in draws:
            draws[b] = bool(rng.random() < 0.5)
        return np.full(n, draws[b])
    return f


def assign_cluster(n_regions: int, seed: int = 0):
    """Geo: each region is permanently treated or control.

    Interference is removed BETWEEN regions (couriers do not cross) but the
    effective sample size collapses from customers to regions, which is the
    trade this design makes.
    """
    rng = np.random.default_rng(seed)
    treated_regions = rng.random(n_regions) < 0.5
    if treated_regions.all() or (~treated_regions).all():   # degenerate draw
        treated_regions[0] = not treated_regions[0]

    def f(t, r, n, rng_unused):
        return np.full(n, bool(treated_regions[r]))
    f.treated_regions = treated_regions
    return f


# ----------------------------------------------------------------------
def validate(mk: Marketplace) -> dict:
    """Stylised facts. A simulator presented as reality is a red flag; a
    simulator whose behaviour is checked and reported is a lab."""
    busy = np.linspace(0, mk.couriers[0] * 0.99, 25)
    etas = np.array([float(mk.eta(np.full(mk.R, b))[0]) for b in busy])
    rising = bool(np.all(np.diff(etas) > 0))
    convex = bool(etas[-1] - etas[-2] > etas[1] - etas[0])
    conv_falls = bool(mk.conversion(np.array([60.0]), np.array([False]), 0)[0]
                      < mk.conversion(np.array([18.0]), np.array([False]), 0)[0])
    h = np.arange(0, 1440)
    d = demand_curve(h)
    peaks_ok = bool(d[12 * 60:13 * 60].mean() > d[3 * 60:4 * 60].mean()
                    and d[18 * 60:19 * 60].mean() > d[3 * 60:4 * 60].mean())
    return dict(eta_rises_with_utilisation=rising,
                eta_is_convex_in_utilisation=convex,
                conversion_falls_with_eta=conv_falls,
                demand_has_meal_peaks=peaks_ok)
