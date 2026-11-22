"""
pe_tools.py: helper tools for the CHEM 73000 Lecture 5 tutorial
"Polydisperse polyethylene melt with G2RINS + MARTINI 3".

What is in here
---------------
1. sample_chains()      G2RINS string  ->  an ensemble of polyethylene chains
2. martini3_beads()     carbons per chain  ->  MARTINI 3 beads per chain (4:1 mapping)
3. write_topology()     one GROMACS .itp molecule type per chain length + topol.top
4. build_melt()         ideal random-walk chains packed at melt density (.gro)
5. load_run(), chain_com(), msd_fft(), fit_diffusion(), diffusion_by_length(),
   rg2_time(), rg_by_chain()   analysis of the trajectory (MSD, diffusion, size)
6. show_melt(), show_mapping(), animate_chains(), render_melt_png()
                        3D visualization (py3Dmol) and a static "poster" render

The MARTINI 3 polyethylene model
--------------------------------
Every bead is a regular MARTINI 3 "C1" bead standing for 4 CH2 groups (C4H8),
exactly like the linear alkanes in the official MARTINI 3 solvent library
(octane = 2 C1, dodecane = 3 C1, hexadecane = 4 C1; Souza et al., Nat. Methods
18, 382, 2021; martini_v3.0.0_solvents_v1.itp). We use the same bonded
parameters for chains of any length:
    bonds   b0 = 0.475 nm (chain-end bonds), 0.47 nm (inner bonds), kb = 3800 kJ mol-1 nm-2
    angles  theta0 = 180 deg, k_theta = 35 kJ mol-1 (GROMACS angle function 2)
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
CARBONS_PER_BEAD = 4          # 4:1 mapping, one C1 bead = C4H8
BOND_END = 0.475              # nm, as in MARTINI 3 octane / hexadecane
BOND_MID = 0.470              # nm
K_BOND = 3800.0               # kJ mol-1 nm-2
ANGLE_0 = 180.0               # deg
K_ANGLE = 35.0                # kJ mol-1, GROMACS angle type 2 (cosine-harmonic)
BEAD_MASS_REAL = 56.11        # g/mol of C4H8 (MARTINI itself uses 72 for C1)
PE_MELT_DENSITY = 0.78        # g/cm3, polyethylene melt near 450 K (used only to size the box)
AVOGADRO = 6.02214076e23


# ----------------------------------------------------------------------------
# 1. G2RINS -> chains
# ----------------------------------------------------------------------------
def sample_chains(g2rins_string: str, n_chains: int = 200, seed: int = 2026) -> dict:
    """Generate `n_chains` chains from a G2RINS string with the g2rins package.

    Returns a dict with the SMILES of every chain, its number of carbon atoms,
    its molecular weight (g/mol) and Mn, Mw and dispersity of the sample.
    """
    import g2rins
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    creator = g2rins.G2rins.make(g2rins_string).get_graph_creator().get_ensemble_creator()
    ens = creator.create_ensemble(n_chains, output_format="smiles", ensemble_info=True, seed=seed)
    if ens is None:
        raise RuntimeError("G2RINS could not sample this string; check it in the G2RINS hands-on notebook.")
    smiles = list(ens.chains)
    n_c, mw = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        n_c.append(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6))
        molh = Chem.AddHs(mol)
        mw.append(sum(a.GetMass() for a in molh.GetAtoms()))
    n_c, mw = np.array(n_c), np.array(mw)
    mn = mw.mean()
    mw_avg = (mw ** 2).sum() / mw.sum()
    return dict(smiles=smiles, n_carbons=n_c, mw=mw, Mn=mn, Mw=mw_avg, D=mw_avg / mn,
                g2rins=g2rins_string, seed=seed)


# ----------------------------------------------------------------------------
# 2. carbons -> MARTINI 3 beads
# ----------------------------------------------------------------------------
def martini3_beads(n_carbons) -> np.ndarray:
    """4 carbons -> 1 C1 bead (rounded). Chains shorter than octane become 2 beads."""
    n_carbons = np.asarray(n_carbons)
    return np.maximum(2, np.rint(n_carbons / CARBONS_PER_BEAD).astype(int))


# ----------------------------------------------------------------------------
# 3. topology
# ----------------------------------------------------------------------------
def _moltype_block(n: int) -> str:
    name = f"PE{n}"
    lines = [f"[ moleculetype ]", "; name  nrexcl", f"  {name}  1", "",
             "[ atoms ]", "; id type resnr residue atom cgnr charge"]
    for i in range(1, n + 1):
        lines.append(f"  {i:4d}  C1  1  PE  C{i}  {i:4d}  0")
    lines += ["", "[ bonds ]", "; i  j  funct  length  kb"]
    for i in range(1, n):
        b0 = BOND_END if (i == 1 or i + 1 == n) else BOND_MID
        lines.append(f"  {i:4d} {i + 1:4d}  1  {b0:.3f}  {K_BOND:.0f}")
    if n >= 3:
        lines += ["", "[ angles ]", "; i  j  k  funct  angle  k_theta"]
        for i in range(1, n - 1):
            lines.append(f"  {i:4d} {i + 1:4d} {i + 2:4d}  2  {ANGLE_0:.1f}  {K_ANGLE:.1f}")
    return "\n".join(lines) + "\n"


def write_topology(n_beads, outdir: str, title: str = "Polydisperse polyethylene melt (MARTINI 3)",
                   forcefield: str = "martini_v3.0.0_C1.itp") -> list:
    """Write PE_chains.itp (one molecule type per chain length) and topol.top.

    Chains are written longest first; returns that chain order (bead counts) so the
    coordinates can be written in the same order.
    """
    os.makedirs(outdir, exist_ok=True)
    order = sorted((int(n) for n in n_beads), reverse=True)
    counts = Counter(order)
    lengths = sorted(counts, reverse=True)
    with open(os.path.join(outdir, "PE_chains.itp"), "w") as f:
        f.write(";;; MARTINI 3 polyethylene, one C1 bead = 4 CH2 (see pe_tools.py for the parameters)\n\n")
        for n in lengths:
            f.write(_moltype_block(n) + "\n")
    with open(os.path.join(outdir, "topol.top"), "w") as f:
        f.write(f'#include "{forcefield}"\n#include "PE_chains.itp"\n\n')
        f.write(f"[ system ]\n{title}\n\n[ molecules ]\n; name  number\n")
        for n in lengths:
            f.write(f"PE{n}  {counts[n]}\n")
    return order


# ----------------------------------------------------------------------------
# 4. initial coordinates: random-walk chains in a box
# ----------------------------------------------------------------------------
def build_melt(chain_order, outfile: str, fill_fraction: float = 1.0, seed: int = 2026,
               min_dist: float = 0.20, temperature: float = 450.0) -> float:
    """Pack the chains in a periodic cubic box at melt density, as ideal (random-walk) chains.

    In a melt every chain is an ideal random walk (Flory), so each chain is grown with the
    bending-angle distribution of the MARTINI 3 angle potential at `temperature`
    (P ~ exp(-V/kT) with V = k/2 (1 - cos psi)^2, psi = 180 deg - angle) and no long-range
    self-avoidance. New beads only keep `min_dist` from beads already placed; the remaining
    overlaps are pushed apart by the energy minimization. Building directly at melt density
    (fill_fraction = 1) avoids squeezing the chains later, which long chains would need
    hundreds of nanoseconds to undo. Returns the box edge length (nm).
    """
    rng = np.random.default_rng(seed)
    n_total = int(np.sum(chain_order))
    vol_per_bead = BEAD_MASS_REAL / (PE_MELT_DENSITY * AVOGADRO) * 1e21          # nm3
    L = (n_total * vol_per_bead / fill_fraction) ** (1 / 3)
    ncell = max(1, int(L // max(min_dist, 0.25)))
    cell = L / ncell
    grid: dict = {}
    a = 0.5 * K_ANGLE / (0.0083145 * temperature)       # exp(-a x^2), x = 1 - cos(psi)

    def cell_of(p):
        return tuple((np.floor((p % L) / cell).astype(int)) % ncell)

    def free(p):
        c = cell_of(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in grid.get(((c[0] + dx) % ncell, (c[1] + dy) % ncell, (c[2] + dz) % ncell), ()):
                        d = p - q
                        d -= L * np.rint(d / L)
                        if d @ d < min_dist ** 2:
                            return False
        return True

    def put(p):
        grid.setdefault(cell_of(p), []).append(p % L)

    def bend(u):
        """New unit bond vector at a Boltzmann-distributed angle psi from u."""
        while True:
            x = abs(rng.normal(0.0, np.sqrt(0.5 / a)))
            if x <= 2.0:
                break
        cpsi = 1.0 - x
        spsi = np.sqrt(max(0.0, 1.0 - cpsi ** 2))
        w = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
        w /= np.linalg.norm(w)
        v = np.cross(u, w)
        phi = rng.uniform(0, 2 * np.pi)
        return cpsi * u + spsi * (np.cos(phi) * w + np.sin(phi) * v)

    coords = []
    for n in chain_order:
        for _attempt in range(500):
            chain = [rng.random(3) * L]
            if not free(chain[0]):
                continue
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            ok = True
            for i in range(1, n):
                b0 = BOND_END if (i == 1 or i == n - 1) else BOND_MID
                for _try in range(100):
                    step = bend(u) if i > 1 else u
                    p = chain[-1] + b0 * step
                    if free(p) and all(np.sum((p - q) ** 2) > min_dist ** 2 for q in chain[:-2]):
                        chain.append(p)
                        u = step
                        break
                    if i == 1:
                        u = rng.normal(size=3)
                        u /= np.linalg.norm(u)
                else:
                    ok = False
                    break
            if ok:
                for p in chain:
                    put(p)
                coords.append(np.array(chain))
                break
        else:
            raise RuntimeError("Could not place a chain; lower fill_fraction.")

    with open(outfile, "w") as f:
        f.write("Polydisperse polyethylene melt, MARTINI 3 (built with pe_tools.py)\n")
        f.write(f"{n_total:5d}\n")
        atom = 0
        for ic, chain in enumerate(coords, 1):
            for ib, (x, y, z) in enumerate(chain, 1):
                atom += 1
                f.write(f"{ic % 100000:5d}{'PE':<5s}{('C' + str(ib))[:5]:>5s}{atom % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"{L:10.5f}{L:10.5f}{L:10.5f}\n")
    return L


def wrap_gro(src: str, dst: str, margin: float = 0.001) -> str:
    """Copy a .gro file with every bead put inside the periodic box (velocities kept).

    GROMACS versions before 2023.5 can garble the run input (.tpr) when beads sit outside the
    box during grompp's pair-list buffer estimate (GROMACS issue 5002). Chains built by
    build_melt and structures written by mdrun keep molecules whole, so wrap them first.
    """
    lines = open(src).read().splitlines()
    box = [float(v) for v in lines[-1].split()[:3]]
    out = lines[:2]
    for ln in lines[2:-1]:
        xyz = []
        for k, length in enumerate(box):
            v = float(ln[20 + 8 * k: 28 + 8 * k]) % length
            xyz.append(min(v, length - margin))
        out.append(ln[:20] + "".join(f"{v:8.3f}" for v in xyz) + ln[44:])
    out.append(lines[-1])
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    return dst


def check_tpr(tpr: str) -> int:
    """Check that a run input only bonds neighbouring beads (i, i+1) and angles (i, i+1, i+2).

    Returns the number of bonds plus angles; raises RuntimeError if the topology is garbled.
    """
    import re
    import subprocess
    dump = subprocess.run(["gmx", "dump", "-s", tpr], capture_output=True, text=True).stdout.split("\nx (")[0]
    n = bad = 0
    for m in re.finditer(r"type=\d+ \((BONDS|G96ANGLES)\)\s+([\d ]+)", dump):
        a = [int(v) for v in m.group(2).split()]
        want = 2 if m.group(1) == "BONDS" else 3
        n += 1
        bad += not (len(a) == want and all(a[i + 1] == a[i] + 1 for i in range(want - 1)))
    if n == 0 or bad:
        raise RuntimeError(f"{tpr}: {bad} garbled bonded interactions; rerun grompp on a structure from wrap_gro().")
    return n


# ----------------------------------------------------------------------------
# 5. analysis
# ----------------------------------------------------------------------------
def load_run(gro: str, xtc=None):
    """MDAnalysis Universe (one residue per chain, chain bonds added).

    gro: the start.gro written by build_melt (it numbers the chains);
    xtc: an unwrapped trajectory (gmx trjconv -pbc nojump), a list of trajectory pieces
         that follow each other in time, or None for a single structure.
    """
    import warnings
    import MDAnalysis as mda
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                    # "Reader has no dt information" for .gro
        u = mda.Universe(gro, xtc) if xtc else mda.Universe(gro)
    bonds = []
    for res in u.residues:
        idx = res.atoms.indices
        bonds += list(zip(idx[:-1], idx[1:]))
    u.add_TopologyAttr("bonds", bonds)
    if xtc:
        broken = max_bond_length(u)
        if broken > 0.25 * u.dimensions[:3].min() / 10.0:
            raise ValueError(
                f"Some chains are split across the periodic box (longest bond {broken:.1f} nm). "
                "Unwrap with a whole reference structure, e.g.\n"
                "  echo 0 | gmx trjconv -s equil.gro -f md.xtc -o md_nojump.xtc -pbc nojump")
    return u


def max_bond_length(u, frame: int = 0) -> float:
    """Longest bead-bead bond (nm) in one frame; about 0.5 nm when every chain is whole."""
    u.trajectory[frame]
    sizes, starts = chain_info(u)
    x = u.atoms.positions / 10.0
    b = np.linalg.norm(np.diff(x, axis=0), axis=1)
    inner = np.ones(len(b), bool)
    inner[starts[1:] - 1] = False                          # skip the gap between consecutive chains
    return float(b[inner].max())


def chain_info(u):
    """Bead count of every chain and the atom-index slices of each chain."""
    sizes = np.array([len(r.atoms) for r in u.residues])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return sizes, starts


def chain_com(u, step: int = 1):
    """Centre-of-mass trajectory of every chain (all C1 beads have the same mass).

    Returns times (ns, counted from the first frame, so they double as MSD lag times) and a
    com array (n_frames, n_chains, 3) in nm with the drift of the whole box removed.
    Use an unwrapped trajectory (gmx trjconv -pbc nojump, see load_run).
    """
    sizes, starts = chain_info(u)
    t, com = [], []
    for ts in u.trajectory[::step]:
        x = u.atoms.positions / 10.0                       # Angstrom -> nm
        c = np.add.reduceat(x, starts, axis=0) / sizes[:, None]
        com.append(c)
        t.append(ts.time / 1000.0)                         # ps -> ns
    com, t = np.array(com), np.array(t)
    drift = (com * sizes[None, :, None]).sum(axis=1) / sizes.sum()
    return t - t[0], com - drift[:, None, :]


def bead_positions(u, which: str = "middle", step: int = 1):
    """Trajectory of one bead per chain ('middle' bead, or 'end'), drift removed; times from the first frame."""
    sizes, starts = chain_info(u)
    pick = starts + sizes // 2 if which == "middle" else starts
    pos, t = [], []
    for ts in u.trajectory[::step]:
        pos.append(u.atoms.positions[pick] / 10.0)
        t.append(ts.time / 1000.0)
    pos, t = np.array(pos), np.array(t)
    return t - t[0], pos - pos.mean(axis=1, keepdims=True)


def msd_fft(x):
    """Mean-squared displacement averaged over all time origins (FFT algorithm).

    x: array (n_frames, n_particles, 3). Returns msd[lag, particle] in the units of x squared.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    # S2 term: autocorrelation by FFT
    f = np.fft.rfft(x, n=2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n].sum(axis=2)
    acf /= (n - np.arange(n))[:, None]
    # S1 term: recursive sum of squares
    d = (x ** 2).sum(axis=2)
    d = np.vstack([d, np.zeros((1, d.shape[1]))])
    q = 2 * d.sum(axis=0)
    s1 = np.zeros((n, x.shape[1]))
    for m in range(n):
        q = q - d[m - 1] - d[n - m]
        s1[m] = q / (n - m)
    return s1 - 2 * acf


def fit_diffusion(t, msd, t_min=None, t_max=None):
    """Einstein relation MSD = 6 D t + c, fitted over [t_min, t_max] (ns).

    Default window: 10 % to 40 % of the trajectory (many time origins, diffusive regime).
    Returns D in nm2/ns (multiply by 1e-5 for cm2/s).
    """
    t = np.asarray(t)
    t_min = 0.10 * t[-1] if t_min is None else t_min
    t_max = 0.40 * t[-1] if t_max is None else t_max
    m = (t >= t_min) & (t <= t_max)
    slope, intercept = np.polyfit(t[m], msd[m], 1)
    return slope / 6.0


def local_slope(t, msd, n_points: int = 40, t_max_frac: float = 0.5):
    """d ln(MSD) / d ln(t) on log-spaced lag times: 2 = ballistic, ~0.5 = Rouse, 1 = free diffusion.

    Uses lags up to t_max_frac of the run (longer lags have too few time origins).
    Returns the mid-point lag times and the slopes.
    """
    t, msd = np.asarray(t), np.asarray(msd)
    keep = np.where((t > 0) & (t <= t_max_frac * t[-1]))[0]
    pick = keep[np.unique(np.round(np.logspace(0, np.log10(len(keep)), n_points)).astype(int) - 1)]
    lt, lm = np.log(t[pick]), np.log(msd[pick])
    return np.exp(0.5 * (lt[1:] + lt[:-1])), np.diff(lm) / np.diff(lt)


def diffusion_by_length(t, com_msd, sizes, bins=None, n_boot=300, seed=1, rg2=None, **fit_kw):
    """Diffusion coefficient of chains grouped by length, with bootstrap error bars.

    Returns a list of dicts: N range, mean N, number of chains, D and its 16-84 % interval.
    If rg2 (per-chain squared radius of gyration) is given, 'diffusive' tells whether the
    chains moved farther than their own size, MSD > 2 <Rg^2>, by the end of the fit window;
    otherwise the chain has not yet entered free diffusion and D is only an upper estimate.
    """
    rng = np.random.default_rng(seed)
    if bins is None:
        bins = [2, 3, 5, 9, 17, 33, 65, 129, 257, 10 ** 6]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = np.where((sizes >= lo) & (sizes < hi))[0]
        if len(sel) < 3:
            continue
        D = fit_diffusion(t, com_msd[:, sel].mean(axis=1), **fit_kw)
        boots = [fit_diffusion(t, com_msd[:, rng.choice(sel, len(sel))].mean(axis=1), **fit_kw)
                 for _ in range(n_boot)]
        row = dict(N_lo=lo, N_hi=hi - 1, N=sizes[sel].mean(), chains=len(sel), D=D,
                   D_lo=np.percentile(boots, 16), D_hi=np.percentile(boots, 84))
        if rg2 is not None:
            t_max = fit_kw.get("t_max") or 0.40 * np.asarray(t)[-1]
            k = np.searchsorted(t, t_max)
            row["msd_end"] = com_msd[min(k, len(t) - 1), sel].mean()
            row["rg2"] = np.mean(np.asarray(rg2)[sel])
            row["diffusive"] = bool(row["msd_end"] > 2 * row["rg2"])
        out.append(row)
    return out


def plot_com_paths(t, com, sizes, chains, ax=None, cmap="plasma"):
    """3D paths of the centres of mass of selected chains (short chains wander far, long ones crawl)."""
    import matplotlib.pyplot as plt
    colors, _, _ = _chain_colors(sizes, cmap)
    if ax is None:
        fig = plt.figure(figsize=(7, 6), facecolor="#0b1020")
        ax = fig.add_subplot(projection="3d", facecolor="#0b1020")
    for c in chains:
        p = com[:, c] - com[0, c]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=colors[c], lw=1.6, alpha=0.95, label=f"N = {sizes[c]}")
        ax.scatter(*p[-1], color=colors[c], s=40, edgecolors="white", linewidths=0.6)
    ax.scatter(0, 0, 0, color="white", s=30)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.04, 0.06, 0.13, 1.0))
        axis.label.set_color("white")
        axis.set_tick_params(colors="#9aa4b2")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)"); ax.set_zlabel("z (nm)")
    leg = ax.legend(loc="upper left", fontsize=8, facecolor="#0b1020", edgecolor="#334155")
    for txt in leg.get_texts():
        txt.set_color("white")
    ax.set_title(f"Centre-of-mass paths over {t[-1]:.0f} ns (all start at the white dot)", color="white")
    return ax


def rg2_time(u, step: int = 5):
    """Squared radius of gyration (nm2) of every chain in every `step`-th frame.

    Returns times (ns, from the first frame), the bead count of every chain, and an array
    (n_frames, n_chains). Flat curves for every chain length mean the melt is equilibrated.
    """
    sizes, starts = chain_info(u)
    out, t = [], []
    for ts in u.trajectory[::step]:
        x = u.atoms.positions / 10.0
        com = np.add.reduceat(x, starts, axis=0) / sizes[:, None]
        d2 = ((x - np.repeat(com, sizes, axis=0)) ** 2).sum(axis=1)
        out.append(np.add.reduceat(d2, starts) / sizes)
        t.append(ts.time / 1000.0)
    t = np.array(t)
    return t - t[0], sizes, np.array(out)


def bond_statistics(u, step: int = 20):
    """Mean bond length b (nm) and mean cosine <cos psi> between consecutive bond vectors."""
    sizes, starts = chain_info(u)
    first = np.zeros(int(sizes.sum()) - 1, bool)          # bond k joins bead k and k+1 of the same chain
    first[:] = True
    first[starts[1:] - 1] = False
    b_all, c_all = [], []
    for ts in u.trajectory[::step]:
        x = u.atoms.positions / 10.0
        v = np.diff(x, axis=0)
        bl = np.linalg.norm(v, axis=1)
        b_all.append(bl[first].mean())
        nxt = first[:-1] & first[1:]                       # two bonds in a row inside one chain
        c_all.append(np.mean(np.sum(v[:-1][nxt] * v[1:][nxt], axis=1) / (bl[:-1][nxt] * bl[1:][nxt])))
    return float(np.mean(b_all)), float(np.mean(c_all))


def frc_rg2(n_beads, b: float, c: float):
    """<Rg^2> (nm2) of an ideal freely rotating chain: bond length b, <cos psi> = c between neighbouring bonds."""
    out = []
    for N in np.atleast_1d(n_beads):
        k = np.arange(1, int(N))
        r2 = b * b * (k * (1 + c) / (1 - c) - 2 * c * (1 - c ** k) / (1 - c) ** 2)
        out.append(np.sum((N - k) * r2) / N ** 2)
    return np.array(out)


def rg_by_chain(u, step: int = 5):
    """Time-averaged squared radius of gyration (nm2) of every chain; returns (sizes, rg2)."""
    _, sizes, rg2 = rg2_time(u, step)
    return sizes, rg2.mean(axis=0)


# ----------------------------------------------------------------------------
# 6. visualization
# ----------------------------------------------------------------------------
def _chain_colors(sizes, cmap="plasma"):
    """One color per chain from its length (log scale). Returns hex colors, the norm and the colormap used."""
    import matplotlib
    import matplotlib.colors as mcolors
    sizes = np.asarray(sizes)
    norm = mcolors.LogNorm(vmin=max(2, sizes.min()), vmax=max(sizes.max(), 3))
    cm = mcolors.ListedColormap(matplotlib.colormaps[cmap](np.linspace(0.08, 0.92, 256)))
    return [mcolors.to_hex(cm(norm(s))) for s in sizes], norm, cm


def _whole_chains_in_box(u):
    """Coordinates (Angstrom) with every chain whole and its centre inside the box."""
    sizes, starts = chain_info(u)
    x = u.atoms.positions.copy()
    L = u.dimensions[:3]
    com = np.add.reduceat(x, starts, axis=0) / sizes[:, None]
    shift = np.floor(com / L) * L
    return x - np.repeat(shift, sizes, axis=0)


def _pdb_block(x, sizes, chains=None, box=None):
    """PDB text, one residue per chain, bonds as CONECT records.

    If box (Angstrom) is given the coordinates are wrapped, so bonds that cross the box
    boundary are left out instead of being drawn straight across the box.
    """
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    chains = range(len(sizes)) if chains is None else chains
    lines, conect, serial = [], [], 0
    for c in chains:
        first = serial + 1
        for k in range(sizes[c]):
            serial += 1
            px, py, pz = x[starts[c] + k]
            lines.append(f"HETATM{serial:5d}  C   PE  A{(c + 1) % 10000:4d}    {px:8.3f}{py:8.3f}{pz:8.3f}  1.00  0.00           C")
        for a in range(first, serial):
            if box is not None:
                d = np.abs(x[starts[c] + a - first + 1] - x[starts[c] + a - first])
                if np.any(d > 0.5 * box):
                    continue
            conect.append(f"CONECT{a:5d}{a + 1:5d}")
    return "\n".join(lines + conect + ["END"])


def show_melt(u, frame: int = -1, cmap: str = "plasma", highlight=None, spin: bool = False, wrap=None,
              width: int = 760, height: int = 520, background: str = "#0b1020"):
    """Interactive 3D view of the melt: every chain colored by its length (purple = short, yellow = long).

    highlight: list of chain indices to show bright while the rest of the melt is ghosted.
    wrap: True puts every bead inside the box (a full cube, the classic MD snapshot);
          False keeps every chain whole (chains stick out of the box). Default: wrap unless highlighting.
    """
    import py3Dmol
    u.trajectory[frame]
    sizes, _ = chain_info(u)
    colors, _, _ = _chain_colors(sizes, cmap)
    L = u.dimensions[:3]
    highlight = None if highlight is None else [int(c) for c in highlight]
    wrap = (highlight is None) if wrap is None else wrap
    if highlight is None:
        block = _pdb_block(u.atoms.positions % L, sizes, box=L) if wrap else _pdb_block(_whole_chains_in_box(u), sizes)
    view = py3Dmol.view(width=width, height=height)
    if highlight is None:
        view.addModel(block, "pdb")
        for c, col in enumerate(colors):
            view.setStyle({"model": -1, "resi": int(c + 1) % 10000},
                          {"sphere": {"color": col, "radius": 2.1}, "stick": {"color": col, "radius": 0.9}})
    else:
        x_all = u.atoms.positions % L if wrap else _whole_chains_in_box(u)
        box = L if wrap else None
        rest = [c for c in range(len(sizes)) if c not in set(highlight)]
        view.addModel(_pdb_block(x_all, sizes, rest, box=box), "pdb")              # model 0: the crowd
        view.setStyle({"model": 0}, {"sphere": {"color": "#64748b", "radius": 1.5, "opacity": 0.16}})
        view.addModel(_pdb_block(x_all, sizes, list(highlight), box=box), "pdb")   # model 1: chosen chains
        for c in highlight:
            col = colors[c]
            view.setStyle({"model": 1, "resi": int(c + 1) % 10000},
                          {"sphere": {"color": col, "radius": 2.3}, "stick": {"color": col, "radius": 1.0}})
    Lx, Ly, Lz = (float(v) for v in L)                      # plain floats: py3Dmol sends JSON
    view.addBox({"center": {"x": Lx / 2, "y": Ly / 2, "z": Lz / 2},
                 "dimensions": {"w": Lx, "h": Ly, "d": Lz},
                 "wireframe": True, "color": "#7dd3fc"})
    view.setBackgroundColor(background)
    view.setViewStyle({"style": "outline", "color": "#05070d", "width": 0.04})
    view.zoomTo()
    view.rotate(-25, "x")
    view.rotate(30, "y")
    view.zoom(0.9)
    if spin:
        view.spin("y", 0.6)
    return view


def animate_chains(u, chains, stride: int = 5, cmap: str = "plasma", width: int = 760, height: int = 520,
                   background: str = "#0b1020"):
    """Movie of selected chains moving through the melt (unwrapped coordinates)."""
    import py3Dmol
    chains = [int(c) for c in chains]
    sizes, starts = chain_info(u)
    colors, _, _ = _chain_colors(sizes, cmap)
    models = []
    for ts in u.trajectory[::stride]:
        models.append(_pdb_block(u.atoms.positions, sizes, chains))
    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames("\n".join(m.replace("END", "ENDMDL") for m in models), "pdb")
    for c in chains:
        view.setStyle({"resi": int(c + 1) % 10000}, {"sphere": {"color": colors[c], "radius": 2.0},
                                                  "stick": {"color": colors[c], "radius": 0.9}})
    view.setBackgroundColor(background)
    view.zoomTo()
    view.animate({"loop": "forward", "interval": 80})
    return view


def show_mapping(aa_gro: str, chain: int = 0, atoms_per_chain: int = 62, width: int = 720, height: int = 420):
    """One all-atom C20 chain (sticks) with its five MARTINI 3 beads drawn on top (4 carbons per bead)."""
    import MDAnalysis as mda
    import py3Dmol
    u = mda.Universe(aa_gro)
    ag = u.atoms[chain * atoms_per_chain:(chain + 1) * atoms_per_chain]
    carbons = ag.select_atoms("name C*")
    view = py3Dmol.view(width=width, height=height)
    lines = []
    for i, a in enumerate(ag, 1):
        el = "C" if a.name.startswith("C") else "H"
        x, y, z = a.position
        lines.append(f"ATOM  {i:5d} {a.name:<4s} ETH A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {el}")
    view.addModel("\n".join(lines) + "\nEND", "pdb", {"keepH": True})            # 3Dmol drops H unless asked
    view.setStyle({"elem": "C"}, {"stick": {"radius": 0.17, "color": "#cbd5e1"}, "sphere": {"scale": 0.24, "color": "#cbd5e1"}})
    view.setStyle({"elem": "H"}, {"stick": {"radius": 0.17, "color": "#ffffff"}, "sphere": {"scale": 0.16, "color": "#ffffff"}})
    palette = ["#f97316", "#facc15", "#22c55e", "#38bdf8", "#a855f7", "#ec4899"]
    cpos = carbons.positions
    for b in range(len(cpos) // CARBONS_PER_BEAD):
        c = cpos[b * CARBONS_PER_BEAD:(b + 1) * CARBONS_PER_BEAD].mean(axis=0)
        view.addSphere({"center": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}, "radius": 2.35,
                        "color": palette[b % len(palette)], "alpha": 0.55})
    view.setBackgroundColor("#0b1020")
    view.zoomTo()
    return view


def render_melt_png(u, filename: str, frame: int = -1, cmap: str = "plasma", dpi: int = 200, wrap: bool = True,
                    title: str = "Polydisperse polyethylene melt · MARTINI 3", view=(32, 22)):
    """Static 'poster' render (no browser needed): perspective, depth fog, bonds and the box.

    wrap=True fills the periodic box with beads (bonds that cross the box edge are not drawn);
    wrap=False keeps every chain whole. Beads are colored by chain length.
    view = (turn about the vertical axis, tilt toward the viewer) in degrees. Returns the file name.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb

    bg = "#0b1020"
    u.trajectory[frame]
    sizes, starts = chain_info(u)
    colors, norm, cm = _chain_colors(sizes, cmap)
    L = u.dimensions[:3] / 10.0
    x = ((u.atoms.positions / 10.0) % L if wrap else _whole_chains_in_box(u) / 10.0) - L / 2
    th, ph = np.radians(view[0]), np.radians(view[1])
    ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
    R = rx @ ry
    f = 3.2 * L.max()                                             # camera distance (perspective)

    def project(p):
        q = p @ R.T
        s_ = f / (f - q[..., 2])
        return q[..., 0] * s_, q[..., 1] * s_, q[..., 2], s_

    px, py, pz, ps = project(x)
    zmin, zmax = pz.min(), pz.max()
    depth = (pz - zmin) / (zmax - zmin)                            # 0 = back, 1 = front
    rgb = np.repeat(np.array([to_rgb(c) for c in colors]), sizes, axis=0)
    fog = np.array(to_rgb(bg))

    def shade(c, d):                                               # mix toward the background at the back
        w = (0.25 + 0.75 * d)[:, None]
        return np.clip(w * c + (1 - w) * fog, 0, 1)

    # bonds (segments inside each chain)
    inner = np.ones(len(x) - 1, bool)
    inner[starts[1:] - 1] = False
    inner &= np.all(np.abs(np.diff(x, axis=0)) < 0.5 * L, axis=1)   # skip bonds cut by the box edge
    bi = np.where(inner)[0]
    bdepth = 0.5 * (depth[bi] + depth[bi + 1])

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=bg)
    ax.set_facecolor(bg)
    # box edges
    corners = np.array([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (-0.5, 0.5)]) * L
    cx, cy, _, _ = project(corners)
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(np.abs(corners[a] - corners[b]) > 1e-6) == 1:
                ax.plot([cx[a], cx[b]], [cy[a], cy[b]], color="#7dd3fc", lw=0.8, alpha=0.45, zorder=0)
    # draw back to front in depth slabs so near beads hide far ones
    edges = np.linspace(0, 1, 15)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = k == len(edges) - 2
        m = (bdepth >= lo) & ((bdepth < hi) | last)
        if m.any():
            sb = bi[m]
            seg = np.stack([np.c_[px[sb], py[sb]], np.c_[px[sb + 1], py[sb + 1]]], axis=1)
            ax.add_collection(LineCollection(seg, colors=shade(rgb[sb], bdepth[m]),
                                             linewidths=1.2 + 2.2 * (lo + hi) / 2, zorder=2 * k + 1, capstyle="round"))
        sel = np.where((depth >= lo) & ((depth < hi) | last))[0]
        if len(sel):
            size = (10 + 34 * depth[sel]) * ps[sel] ** 2
            ax.scatter(px[sel], py[sel], s=size, c=shade(rgb[sel], depth[sel]), edgecolors="none", zorder=2 * k + 2)
            ax.scatter(px[sel] - 0.035, py[sel] + 0.035, s=0.18 * size, c="white", alpha=0.10 + 0.30 * depth[sel],
                       edgecolors="none", zorder=2 * k + 2)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=15, pad=14)
    sm = ScalarMappable(norm=norm, cmap=cm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
    cb.set_label("chain length N (MARTINI 3 beads)", color="white")
    cb.outline.set_edgecolor("#334155")
    cb.ax.yaxis.set_tick_params(color="white", which="both")
    plt.setp(cb.ax.get_yticklabels(), color="white")
    n_tot = int(sizes.sum())
    ax.text(0.01, 0.01, f"{len(sizes)} chains · {n_tot} beads · box {L[0]:.1f} nm · t = {u.trajectory.ts.time / 1000:.0f} ns",
            transform=ax.transAxes, color="#9aa4b2", fontsize=9)
    fig.savefig(filename, dpi=dpi, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return filename
