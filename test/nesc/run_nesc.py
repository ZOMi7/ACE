
# NESC check-case checker
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NESC_DIR = Path(__file__).resolve().parent
BINARIES = {"dumb_fc.cpp": REPO_ROOT / "build" / "nesc-headless",
            "schedule_fc.cpp": REPO_ROOT / "build" / "nesc-headless-schedule"}

# must match src/planetary_constants.hpp
EARTH_RATE = 7.292115e-5
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
FT = 0.3048

# compared quantities: NESC column, label, group, smallest tolerance
ATMOS_QUANTITIES = [
    ("altitudeMsl_ft", "altitude (ft)", "translation", 0.01),
    ("latitude_deg", "latitude (deg)", "translation", 1e-7),
    ("longitude_deg", "longitude (deg)", "translation", 1e-7),
    ("feVelocity_ft_s_X", "V north (ft/s)", "translation", 1e-3),
    ("feVelocity_ft_s_Y", "V east (ft/s)", "translation", 1e-3),
    ("feVelocity_ft_s_Z", "V down (ft/s)", "translation", 1e-3),
    ("eulerAngle_deg_Roll", "roll (deg)", "rotation", 0.01),
    ("eulerAngle_deg_Pitch", "pitch (deg)", "rotation", 0.01),
    ("eulerAngle_deg_Yaw", "yaw (deg)", "rotation", 0.01),
    ("bodyAngularRateWrtEi_deg_s_Roll", "p (deg/s)", "rotation", 1e-3),
    ("bodyAngularRateWrtEi_deg_s_Pitch", "q (deg/s)", "rotation", 1e-3),
    ("bodyAngularRateWrtEi_deg_s_Yaw", "r (deg/s)", "rotation", 1e-3),
]
ORBIT_QUANTITIES = [
    ("eiPosition_m_X", "J2000 x (m)", "translation", 1.0),
    ("eiPosition_m_Y", "J2000 y (m)", "translation", 1.0),
    ("eiPosition_m_Z", "J2000 z (m)", "translation", 1.0),
    ("eiVelocity_m_s_X", "J2000 vx (m/s)", "translation", 1e-3),
    ("eiVelocity_m_s_Y", "J2000 vy (m/s)", "translation", 1e-3),
    ("eiVelocity_m_s_Z", "J2000 vz (m/s)", "translation", 1e-3),
    ("eulerAngleWrtEi_rad_Roll", "J2000 roll (rad)", "rotation", 1e-3),
    ("eulerAngleWrtEi_rad_Pitch", "J2000 pitch (rad)", "rotation", 1e-3),
    ("eulerAngleWrtEi_rad_Yaw", "J2000 yaw (rad)", "rotation", 1e-3),
    ("bodyAngularRateWrtEi_rad_s_Roll", "p (rad/s)", "rotation", 1e-6),
    ("bodyAngularRateWrtEi_rad_s_Pitch", "q (rad/s)", "rotation", 1e-6),
    ("bodyAngularRateWrtEi_rad_s_Yaw", "r (rad/s)", "rotation", 1e-6),
]

# NESC body axes as columns in the sim's body frame
NESC_AXES = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0]])


def build(fc):
    binary = BINARIES[fc]
    subprocess.run(["make", "headless", f"FC_SRC=test/nesc/{fc}", f"HEADLESS_TARGET={binary.relative_to(REPO_ROOT)}"],
                   cwd=REPO_ROOT, check=True)
    return binary


def case_fc(case_dir):
    return "schedule_fc.cpp" if (case_dir / "config" / "fc_schedule.txt").exists() else "dumb_fc.cpp"


def sim_setting(case_dir, key):
    # value of a top level key in the case's sim.yaml, None if it isn't set
    m = re.search(rf"^{key}:\s*\"?([^\"#\s]+)", (case_dir / "config" / "sim.yaml").read_text(), re.M)
    return m.group(1) if m else None


def epoch_era(case_dir):
    # earth rotation angle at the case's epoch, same formula as src/sim/src/ephemeris.cpp
    m = re.match(r"(\d+)-(\d+)-(\d+)T(\d+):(\d+):([\d.]+)", sim_setting(case_dir, "epoch") or "2007-11-20T00:00:00")
    y, mo, d, h, mi, sec = (float(g) for g in m.groups())
    if mo <= 2:
        y, mo = y - 1, mo + 12
    a = int(y // 100)
    days = (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (mo + 1)) + d + 2 - a + a // 4 - 1524.5
            + (h + mi / 60.0 + sec / 3600.0) / 24.0 - 2451545.0)
    return 2.0 * math.pi * (0.7790572732640 + 1.00273781191135448 * days)


def read_csv(path):
    # columns as float arrays, anything that isn't a number becomes nan
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for key in rows[0]:
        if not key:
            continue
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        cols[key.strip()] = np.array(vals)
    return cols


def geodetic(x, y, z):
    # ECEF to WGS-84 geodetic latitude (rad), longitude (rad), and height (m)
    e2 = WGS84_F * (2.0 - WGS84_F)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(5):
        N = WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - e2 * N / (N + h)))
    N = WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return lat, np.arctan2(y, x), p / np.cos(lat) - N


def quat_matrix(d):
    # sim body to sim ECI, one 3x3 per row
    w, x, y, z = d["qw"], d["qx"], d["qy"], d["qz"]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def euler_321(C):
    # roll, pitch, yaw (rad) of NESC body axes in a reference frame, from the body to reference matrices
    return (np.arctan2(C[:, 2, 1], C[:, 2, 2]),
            -np.arcsin(np.clip(C[:, 2, 0], -1.0, 1.0)),
            np.arctan2(C[:, 1, 0], C[:, 0, 0]))


def body_rates(d):
    # inertial body rates on NESC's roll, pitch, yaw axes
    return d["wz"], d["wx"], d["wy"]


def orbit_quantities(d, era):
    # the sim's ECI rotated back into J2000
    c, s = math.cos(era), math.sin(era)
    q = {}
    for col, key in (("r", "eiPosition_m_"), ("v", "eiVelocity_m_s_")):
        x, y, z = d[col + "x"], d[col + "y"], d[col + "z"]
        q[key + "X"], q[key + "Y"], q[key + "Z"] = c * x - s * y, s * x + c * y, z

    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    roll, pitch, yaw = euler_321(rz @ quat_matrix(d) @ NESC_AXES)
    p, qr, r = body_rates(d)
    q.update({"eulerAngleWrtEi_rad_Roll": roll, "eulerAngleWrtEi_rad_Pitch": pitch, "eulerAngleWrtEi_rad_Yaw": yaw,
              "bodyAngularRateWrtEi_rad_s_Roll": p, "bodyAngularRateWrtEi_rad_s_Pitch": qr,
              "bodyAngularRateWrtEi_rad_s_Yaw": r})
    return d["t"], q


def sim_quantities(d, earth_rate, round_radius):
    # the sim's ECI state turned into the NESC quantities, over a round earth if round_radius is set
    t = d["t"]
    c, s = np.cos(earth_rate * t), np.sin(earth_rate * t)
    rx, ry, rz = d["rx"], d["ry"], d["rz"]

    # ECI is ECEF at t = 0, rotate back by the earth's spin
    x, y, z = c * rx + s * ry, -s * rx + c * ry, rz
    vx_rel = d["vx"] + earth_rate * ry
    vy_rel = d["vy"] - earth_rate * rx
    vx, vy, vz = c * vx_rel + s * vy_rel, -s * vx_rel + c * vy_rel, d["vz"]

    if round_radius:
        lat, lon, h = np.arctan2(z, np.hypot(x, y)), np.arctan2(y, x), np.sqrt(x * x + y * y + z * z) - round_radius
    else:
        lat, lon, h = geodetic(x, y, z)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)

    # body to NED: ECI -> ECEF -> NED
    zero, one = np.zeros_like(t), np.ones_like(t)
    eci_to_ecef = np.stack([np.stack([c, s, zero], -1), np.stack([-s, c, zero], -1), np.stack([zero, zero, one], -1)], -2)
    ecef_to_ned = np.stack([np.stack([-sl * co, -sl * so, cl], -1),
                            np.stack([-so, co, zero], -1),
                            np.stack([-cl * co, -cl * so, -sl], -1)], -2)
    roll, pitch, yaw = euler_321(ecef_to_ned @ eci_to_ecef @ quat_matrix(d) @ NESC_AXES)
    p, q, r = body_rates(d)

    return t, {
        "altitudeMsl_ft": h / FT,
        "latitude_deg": np.degrees(lat),
        "longitude_deg": np.degrees(lon),
        "feVelocity_ft_s_X": (-sl * co * vx - sl * so * vy + cl * vz) / FT,
        "feVelocity_ft_s_Y": (-so * vx + co * vy) / FT,
        "feVelocity_ft_s_Z": (-cl * co * vx - cl * so * vy - sl * vz) / FT,
        "eulerAngle_deg_Roll": np.degrees(roll),
        "eulerAngle_deg_Pitch": np.degrees(pitch),
        "eulerAngle_deg_Yaw": np.degrees(yaw),
        "bodyAngularRateWrtEi_deg_s_Roll": np.degrees(p),
        "bodyAngularRateWrtEi_deg_s_Pitch": np.degrees(q),
        "bodyAngularRateWrtEi_deg_s_Yaw": np.degrees(r),
    }


def angle_period(key):
    # full turn in the column's units for euler angles, None for anything else
    if "eulerAngle" not in key:
        return None
    return 360.0 if "_deg_" in key else 2.0 * math.pi


def resample(t, vals, grid, period):
    # vals at the grid times, unwrapped first if they are angles so a wrap isn't interpolated through
    if period:
        vals = np.unwrap(vals, period=period)
    return np.interp(grid, t, vals)


def plot_quantity(ax_val, ax_diff, label, grid, ours, nesc_runs, stack):
    # left: every NESC run with ACE dashed on top. right: offsets from the NESC mean
    for n, v in nesc_runs.items():
        ax_val.plot(grid, v, linewidth=0.8, label=n)
    ax_val.plot(grid, ours, "k--", linewidth=1.2, label="ACE")
    ax_val.set_title(label)
    ax_val.legend(fontsize=6)

    mean = stack.mean(axis=0)
    ax_diff.fill_between(grid, stack.min(axis=0) - mean, stack.max(axis=0) - mean, color="tab:blue", alpha=0.25, label="NESC spread")
    ax_diff.plot(grid, ours - mean, "k", linewidth=1.0, label="ACE")
    ax_diff.axhline(0.0, color="gray", linewidth=0.6)
    ax_diff.set_title(f"{label}, minus NESC mean")
    ax_diff.legend(fontsize=6)


def compare(case_dir, binary, plot):
    # runs one case, prints its comparison, returns the groups that failed
    print(f"\n== {case_dir.name}", flush=True)
    subprocess.run([str(binary)], cwd=case_dir, check=True, stdout=subprocess.DEVNULL)

    ours = read_csv(sorted((case_dir / "data").glob("*.csv"))[0])
    if case_dir.parent.name == "orbit":
        quantities = ORBIT_QUANTITIES
        t, q = orbit_quantities(ours, epoch_era(case_dir))
    else:
        quantities = ATMOS_QUANTITIES
        rate = 0.0 if sim_setting(case_dir, "earth_rotation") == "false" else EARTH_RATE
        radius = sim_setting(case_dir, "earth_radius")
        t, q = sim_quantities(ours, rate, float(radius) if radius else None)
    nesc = {p.stem: read_csv(p) for p in sorted(case_dir.glob("*_sim_*.csv"))}
    failed = set()

    # the coarsest NESC grid, inside the span every NESC run and ACE cover
    start = max([c["time"][0] for c in nesc.values()] + [t[0]])
    end = min(c["time"][-1] for c in nesc.values())
    coarsest = max(nesc.values(), key=lambda c: np.median(np.diff(c["time"])))
    step = np.median(np.diff(coarsest["time"]))
    if t[-1] < end - 0.5 * step:
        print(f"   ACE stopped at t = {t[-1]:g} s, NESC runs to {end:g} s")
        failed.add("translation")
    grid = coarsest["time"][(coarsest["time"] >= start - 1e-9) & (coarsest["time"] <= min(end, t[-1]) + 1e-9)]

    # ACE turning to nan fails the case, and only the rows before it compare
    state = np.column_stack([ours[k] for k in ("rx", "ry", "rz", "vx", "vy", "vz", "qw", "wx", "wy", "wz")])
    bad = ~np.isfinite(state).all(axis=1)
    if bad.any():
        print(f"   ACE output turned to nan at t = {t[bad][0]:g} s, only the rows before are compared")
        failed.update(("translation", "rotation"))
        t, q = t[~bad], {k: v[~bad] for k, v in q.items()}
        grid = grid[grid <= t[-1] + 1e-9]

    print(f"   {'quantity':<20}{'max |ACE - NESC mean|':>24}{'tolerance':>14}")
    rows = []
    for key, label, group, floor in quantities:
        # every NESC simulation that logged this quantity, on the grid
        period = angle_period(key)
        runs = {n: resample(c["time"], c[key], grid, period)
                for n, c in nesc.items() if key in c and np.isfinite(c[key]).all()}
        if not runs:
            continue
        ace = resample(t, q[key], grid, period)

        # angles are put on the same turn as the first NESC run
        if period:
            ref = next(iter(runs.values()))
            runs = {n: ref + (v - ref + period / 2) % period - period / 2 for n, v in runs.items()}
            ace = ref + (ace - ref + period / 2) % period - period / 2

        stack = np.array(list(runs.values()))
        diff = np.max(np.abs(ace - stack.mean(axis=0)))
        tol = max(np.max(stack.max(axis=0) - stack.min(axis=0)), floor)
        ok = diff <= tol
        if not ok:
            failed.add(group)
        print(f"   {label:<20}{diff:>24.6g}{tol:>14.6g}   {'ok' if ok else 'FAIL'}")
        rows.append((label, ace, runs, stack))

    print(f"   {'FAIL (' + ', '.join(sorted(failed)) + ')' if failed else 'PASS'}")

    if plot and rows:
        import matplotlib.pyplot as plt
        n_rows = (len(rows) + 1) // 2
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.3 * n_rows), sharex=True, squeeze=False)
        fig.suptitle(case_dir.name)
        for i, (label, ace, runs, stack) in enumerate(rows):
            row, col = divmod(i, 2)
            plot_quantity(axes[row, 2 * col], axes[row, 2 * col + 1], label, grid, ace, runs, stack)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(case_dir / "compare.png", dpi=120)
    return failed


def main():
    args = [a for a in sys.argv[1:] if a != "--plot"]
    cases = sorted(p.parent.parent for p in NESC_DIR.glob("*/*/config/rocket.yaml"))
    if args:
        cases = [c for c in cases if c.name in args]

    plot = "--plot" in sys.argv
    binaries = {fc: build(fc) for fc in sorted({case_fc(c) for c in cases})}
    results = {c.name: compare(c, binaries[case_fc(c)], plot) for c in cases}

    print(f"\n{sum(not f for f in results.values())}/{len(results)} cases pass")
    for group in ("translation", "rotation"):
        print(f"   {group}: {sum(group not in f for f in results.values())}/{len(results)} pass")
    for name, f in results.items():
        if f:
            print(f"   FAIL {name} ({', '.join(sorted(f))})")

    if plot:
        import matplotlib.pyplot as plt
        plt.show()
    sys.exit(1 if any(results.values()) else 0)


if __name__ == "__main__":
    main()
