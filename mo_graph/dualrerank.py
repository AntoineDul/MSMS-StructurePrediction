"""Dual-encoder reranker for MS/MS structure identification: MIST-style spectrum encoder x GINE molecular GNN.

Shared by the training notebook (`dual_encoder_reranker.ipynb`) and the offline Kaggle inference notebook
(`kaggle/casmi26_dual_reranker_inference.ipynb`). Only numpy + torch are needed; RDKit is imported lazily and only
for featurising new SMILES.
"""
import math
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------------------------- chemistry
MONO = {"H": 1.00782503207, "C": 12.0, "N": 14.0030740048, "O": 15.99491461956, "P": 30.97376163,
        "S": 31.97207100, "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473,
        "Si": 27.9769265325, "B": 11.0093054, "Se": 79.9165213, "As": 74.9215965, "Na": 22.98976928,
        "K": 38.96370668, "Li": 7.01600455, "Ca": 39.96259098, "Mg": 23.9850417, "Fe": 55.9349375}
ELEMS = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Si", "B", "Se"]   # MIST-style formula vector
EL_MASS = np.array([MONO[e] for e in ELEMS])
N_EL = len(ELEMS)
ELECTRON, PROTON = 0.00054858, 1.007276467
RDBE_W = np.array([{"C": 1, "Si": 1, "N": .5, "P": .5, "B": .5, "H": -.5, "F": -.5, "Cl": -.5, "Br": -.5,
                    "I": -.5}.get(e, 0) for e in ELEMS])
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
ADDUCT_RE = re.compile(r"\[(\d*)M([^\]]*)\](\d*)([+-])$")


@lru_cache(maxsize=None)
def parse_formula(formula):
    """'C20H17NO6' -> dict of element counts (charges ignored)."""
    counts = defaultdict(int)
    for el, n in FORMULA_RE.findall(str(formula).split("+")[0].split("-")[0]):
        counts[el] += int(n) if n else 1
    return dict(counts)


def formula_mass(formula):
    return sum(MONO[el] * n for el, n in parse_formula(formula).items())


def formula_vector(formula):
    c = parse_formula(formula)
    return np.array([c.get(e, 0) for e in ELEMS], dtype=np.int32)


@lru_cache(maxsize=None)
def parse_adduct(adduct):
    """'[2M+Na]+' -> (n_molecules, signed_charge, mass_offset, ion_shifts) or None if unparseable.

    precursor_mz = (n * M + offset) / |charge|   (same convention as config.ADDUCTS)
    ion_shifts: masses added to a neutral fragment subformula to obtain a fragment ion m/z, with a type id
      1 = (de)protonated fragment, 2 = fragment keeps the adduct ion (Na+, K+, Cl-...), 3 = intrinsically charged
    """
    m = ADDUCT_RE.match(str(adduct).strip())
    if m is None:
        return None
    n = int(m.group(1) or 1)
    charge = int(m.group(3) or 1) * (1 if m.group(4) == "+" else -1)
    offset, ion_shifts = 0.0, []
    for sign, k, grp in re.findall(r"([+-])(\d*)([A-Z][A-Za-z0-9]*)", m.group(2)):
        try:
            gm = formula_mass(grp)
        except KeyError:
            return None
        offset += (1 if sign == "+" else -1) * int(k or 1) * gm
        if sign == "+" and grp in ("Na", "K", "Li", "Cl", "Br"):
            ion_shifts.append((MONO[grp] - np.sign(charge) * ELECTRON, 2))
    offset -= charge * ELECTRON
    pol = 1 if charge > 0 else -1
    shifts = [(pol * PROTON, 1)] + ion_shifts
    if m.group(2) == "":
        shifts.append((-pol * ELECTRON, 3))
    return n, charge, offset, tuple(shifts)


def neutral_mass(precursor_mz, adduct):
    p = parse_adduct(adduct)
    if p is None:
        return np.nan
    n, charge, offset, _ = p
    return (abs(charge) * precursor_mz - offset) / n


@lru_cache(maxsize=8192)
def enumerate_subformulae(fvec_key, max_enum=2_000_000):
    """All chemically plausible subformulae of a formula vector, sorted by neutral mass.

    Mass and ring-plus-double-bond equivalent are additive over elements, so both are built as outer sums;
    element counts are only decoded (mixed-radix) for the subformulae that end up matching a peak."""
    fvec = np.array(fvec_key)
    els = np.nonzero(fvec)[0]
    dims = fvec[els] + 1
    if np.prod(dims.astype(float)) > max_enum:
        return None
    masses, rdbe = np.zeros(1), np.zeros(1)
    for e in els:
        k = np.arange(fvec[e] + 1)
        masses = (masses[:, None] + EL_MASS[e] * k).ravel()
        rdbe = (rdbe[:, None] + RDBE_W[e] * k).ravel()
    keep = np.nonzero((rdbe + 1 >= -0.5) & (masses > 5))[0]
    order = keep[np.argsort(masses[keep], kind="stable")]
    return masses[order], order, els, dims


def annotate(mzs, shifts, fvec, ppm=20.0, abs_tol=0.002):
    """MIST-style peak annotation: closest subformula of `fvec` within tolerance for each peak.

    Returns (subformula counts [P,13] uint8, ppm error [P] int8, ion type [P] int8; type 0 = unannotated)."""
    P = len(mzs)
    sub = np.zeros((P, N_EL), np.uint8); err = np.zeros(P, np.int8); typ = np.zeros(P, np.int8)
    enum = enumerate_subformulae(tuple(int(x) for x in fvec))
    if enum is None or P == 0:
        return sub, err, typ
    masses, flat, els, dims = enum
    best = np.full(P, np.inf)
    for shift, t in shifts:
        target = mzs - shift
        pos = np.clip(np.searchsorted(masses, target), 1, len(masses) - 1)
        left, right = masses[pos - 1], masses[pos]
        j = np.where(np.abs(target - left) <= np.abs(target - right), pos - 1, pos)
        d = np.abs(masses[j] - target)
        tol = np.maximum(ppm * 1e-6 * np.abs(target), abs_tol)
        better = (d <= tol) & (d < best - 1e-9)
        best[better] = d[better]
        if better.any():
            dec = np.stack(np.unravel_index(flat[j[better]], dims), 1)
            tmp = np.zeros((better.sum(), N_EL), np.uint8); tmp[:, els] = dec
            sub[better] = tmp
        typ[better] = t
        err[better] = np.clip(np.round((masses[j[better]] - target[better]) / np.maximum(np.abs(target[better]), 1) * 1e6), -20, 20)
    return sub, err, typ


# --------------------------------------------------------------------------------------------- catalog
ATOM_EL = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B", "Se"]
N_HYB = 6
ATOM_VOCAB = [len(ATOM_EL) + 1, 6, 5, 5, N_HYB + 1, 2, 2]   # element, degree, charge, #H, hybridisation, aromatic, ring
BOND_VOCAB = [4, 2, 2]                                       # type, conjugated, ring


def featurize_catalog(ik14, smiles, fp_bits=1024, fp_radius=2):
    """RDKit graphs + Morgan fingerprints + formula + exact mass for a list of structures (rows that fail are dropped)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    hyb = [Chem.HybridizationType.S, Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
           Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
    bond_t = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
    out = dict(ik14=[], smiles=[], atom_x=[], atom_n=[], bond=[], bond_n=[], fp=[], formula=[], mass=[])
    for ik, smi in zip(ik14, smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        atoms = np.array([[ATOM_EL.index(a.GetSymbol()) if a.GetSymbol() in ATOM_EL else len(ATOM_EL),
                           min(a.GetDegree(), 5), int(np.clip(a.GetFormalCharge(), -2, 2)) + 2, min(a.GetTotalNumHs(), 4),
                           hyb.index(a.GetHybridization()) if a.GetHybridization() in hyb else N_HYB,
                           int(a.GetIsAromatic()), int(a.IsInRing())] for a in mol.GetAtoms()], dtype=np.int8)
        bonds = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            f = [bond_t.get(b.GetBondType(), 0), int(b.GetIsConjugated()), int(b.IsInRing())]
            bonds += [(i, j, *f), (j, i, *f)]
        for k, v in [("ik14", ik), ("smiles", smi), ("atom_x", atoms), ("atom_n", len(atoms)),
                     ("bond", np.array(bonds, dtype=np.int32).reshape(-1, 5)), ("bond_n", 2 * mol.GetNumBonds()),
                     ("fp", np.packbits(gen.GetFingerprintAsNumPy(mol).astype(np.uint8))),
                     ("formula", rdMolDescriptors.CalcMolFormula(mol)), ("mass", Descriptors.ExactMolWt(mol))]:
            out[k].append(v)
    return dict(ik14=np.array(out["ik14"]), smiles=np.array(out["smiles"]), atom_x=np.concatenate(out["atom_x"]),
                atom_n=np.array(out["atom_n"]), bond=np.concatenate(out["bond"]), bond_n=np.array(out["bond_n"]),
                fp=np.stack(out["fp"]), formula=np.array(out["formula"]), mass=np.array(out["mass"]))


_NUM_KEYS, _STR_KEYS = ["atom_x", "atom_n", "bond", "bond_n", "fp", "mass"], ["ik14", "smiles", "formula"]


def save_catalog(G, folder):
    """Numeric arrays -> catalog_arrays.npz, strings -> catalog_meta.parquet (no pickle, portable across numpy versions)."""
    import pandas as pd
    from pathlib import Path
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    small = dict(atom_x=np.int8, atom_n=np.int32, bond=np.int16, bond_n=np.int32, fp=np.uint8, mass=np.float64)
    assert np.abs(G["bond"]).max() < 2 ** 15
    np.savez_compressed(folder / "catalog_arrays.npz", **{k: np.asarray(G[k]).astype(small[k]) for k in _NUM_KEYS})
    pd.DataFrame({k: G[k] for k in _STR_KEYS}).to_parquet(folder / "catalog_meta.parquet", index=False)


def load_catalog(folder):
    import pandas as pd
    from pathlib import Path
    folder = Path(folder)
    G = dict(np.load(folder / "catalog_arrays.npz"))
    meta = pd.read_parquet(folder / "catalog_meta.parquet")
    G.update({k: meta[k].to_numpy(str) for k in _STR_KEYS})
    return G


def _flat_ranges(starts, counts):
    return np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())


def subset_catalog(G, idx):
    """Rows `idx` of a catalog dict (graphs are stored flat, so atoms/bonds are sliced per molecule)."""
    idx = np.asarray(idx)
    a_off = np.concatenate([[0], np.cumsum(G["atom_n"])])[:-1]; b_off = np.concatenate([[0], np.cumsum(G["bond_n"])])[:-1]
    out = {k: np.asarray(G[k])[idx] for k in G if k not in ("atom_x", "bond")}
    out["atom_x"] = G["atom_x"][_flat_ranges(a_off[idx], G["atom_n"][idx])]
    out["bond"] = G["bond"][_flat_ranges(b_off[idx], G["bond_n"][idx])]
    return out


def concat_catalogs(*Gs):
    return {k: np.concatenate([np.asarray(G[k]) for G in Gs]) for k in Gs[0]}


def restrict_catalog(G, masses, ppm=10.0, margin=25):
    """Keep only the rows a pool around these neutral masses can use: the 10 ppm window plus `margin` mass-neighbours
    on each side (what `Scorer.molecule_pool(fill=margin)` may pad with). Pools are unchanged, but a large catalog
    then fits in GPU memory and only the needed graphs are embedded."""
    order = np.argsort(G["mass"], kind="stable"); m = np.asarray(G["mass"])[order]
    masses = np.asarray(masses, float); masses = masses[np.isfinite(masses)]
    lo = np.maximum(np.searchsorted(m, masses * (1 - ppm * 1e-6)) - margin, 0)
    hi = np.minimum(np.searchsorted(m, masses * (1 + ppm * 1e-6), side="right") + margin, len(m))
    d = np.zeros(len(m) + 1, int); np.add.at(d, lo, 1); np.add.at(d, hi, -1)
    return subset_catalog(G, order[np.cumsum(d)[:-1] > 0])


class Catalog:
    """Candidate structures sorted by neutral mass, with compact graph/fingerprint tensors on the device."""

    def __init__(self, G, device):
        order = np.argsort(G["mass"], kind="stable")
        a_off = np.concatenate([[0], np.cumsum(G["atom_n"])]); b_off = np.concatenate([[0], np.cumsum(G["bond_n"])])
        self.ik14, self.smiles, self.formula, self.mass = G["ik14"][order], G["smiles"][order], G["formula"][order], G["mass"][order]
        atom_x = np.concatenate([G["atom_x"][a_off[i]:a_off[i + 1]] for i in order])
        bond = np.concatenate([G["bond"][b_off[i]:b_off[i + 1]] for i in order])
        self.N, self.device = len(order), device
        self.index = {ik: i for i, ik in enumerate(self.ik14)}
        self.formulas, self.fid = np.unique(self.formula, return_inverse=True)
        self.FVEC = np.stack([formula_vector(f) for f in self.formulas])
        t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a).astype(dt)).to(device)
        self.T_ATOM, self.T_BOND = t(atom_x, np.uint8), t(bond, np.int16)   # compact storage, cast after gather
        self.T_AN, self.T_BN = t(G["atom_n"][order], np.int64), t(G["bond_n"][order], np.int64)
        self.T_AOFF, self.T_BOFF = torch.cumsum(self.T_AN, 0) - self.T_AN, torch.cumsum(self.T_BN, 0) - self.T_BN
        self.T_FP = t(G["fp"][order], np.uint8)
        self.BITS = 2 ** torch.arange(7, -1, -1, device=device, dtype=torch.uint8)

    def window(self, nm, ppm=10.0):
        tol = nm * ppm * 1e-6
        return np.searchsorted(self.mass, nm - tol), np.searchsorted(self.mass, nm + tol, side="right")

    def _ranges(self, starts, counts):
        batch = torch.repeat_interleave(torch.arange(len(counts), device=self.device), counts)
        local = torch.arange(int(counts.sum()), device=self.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        return batch, starts[batch] + local

    def graph_batch(self, mol_ids):
        m = torch.as_tensor(mol_ids, device=self.device)
        na, nb = self.T_AN[m], self.T_BN[m]
        abatch, aidx = self._ranges(self.T_AOFF[m], na)
        bbatch, bidx = self._ranges(self.T_BOFF[m], nb)
        node_off = torch.cumsum(na, 0) - na
        b = self.T_BOND[bidx].long()
        return dict(x=self.T_ATOM[aidx].long(), batch=abatch, n_graphs=len(m), src=b[:, 0] + node_off[bbatch],
                    dst=b[:, 1] + node_off[bbatch], battr=b[:, 2:])

    def unpack_fp(self, mol_ids):
        return ((self.T_FP[torch.as_tensor(mol_ids, device=self.device)][..., None] & self.BITS) > 0).flatten(1).float()


# --------------------------------------------------------------------------------------------- spectra
class Spectra:
    """Query spectra as fixed-size arrays. Cleaning matches the hidden test set (peaks above precursor + 2 Da removed),
    then the `max_peaks` most intense peaks, sorted by m/z, intensities scaled to max 1."""

    def __init__(self, df, adduct_vocab, max_peaks=64):
        n, P = len(df), max_peaks
        self.MZ = np.zeros((n, P), np.float32); self.INT = np.zeros((n, P), np.float32); self.NPK = np.zeros(n, np.int32)
        for q, (mz, it, pmz) in enumerate(zip(df["ms2_mzs"], df["ms2_normalized_intensities"], df["precursor_mz"])):
            mz, it = np.asarray(mz, float), np.asarray(it, float)
            k = min(len(mz), len(it)); mz, it = mz[:k], it[:k]
            ok = np.isfinite(mz) & np.isfinite(it) & (mz > 0) & (it > 0) & (mz <= pmz + 2)
            mz, it = mz[ok], it[ok]
            keep = np.sort(np.argsort(it)[-P:]) if len(it) > P else np.arange(len(it))
            keep = keep[np.argsort(mz[keep])]
            self.NPK[q] = len(keep); self.MZ[q, :len(keep)] = mz[keep]
            self.INT[q, :len(keep)] = it[keep] / max(it.max(), 1e-9) if len(it) else 0
        ce = df["collision_energy_ev"]
        self.energy = np.array([float(np.mean(e)) if e is not None and len(e) else np.nan for e in ce])
        self.n_energies = np.array([len(e) if e is not None else 0 for e in ce])
        self.prec = df["precursor_mz"].to_numpy(float)
        self.adduct = df["adduct"].astype(str).to_numpy()
        pol = np.where(np.char.endswith(self.adduct.astype(str), "+"), 1.0, -1.0)
        self.META = np.stack([self.prec / 1000, np.nan_to_num(self.energy) / 100, self.n_energies / 4,
                              np.isfinite(self.energy).astype(float), pol], 1).astype(np.float32)
        vocab = {a: i for i, a in enumerate(adduct_vocab)}
        self.ADDUCT_ID = np.array([vocab.get(a, 0) for a in self.adduct])
        self.SHIFTS = [parse_adduct(a)[3] if parse_adduct(a) else ((PROTON * p, 1),) for a, p in zip(self.adduct, pol)]
        self.neutral = np.array([neutral_mass(m, a) for m, a in zip(self.prec, self.adduct)])


# --------------------------------------------------------------------------------------------- model
def fourier(x, n=16, lo=1.0, hi=2000.0):
    freqs = torch.exp(torch.linspace(math.log(2 * math.pi / hi), math.log(2 * math.pi / lo), n, device=x.device))
    a = x[..., None] * freqs
    return torch.cat([a.sin(), a.cos()], -1)


class PairBiasLayer(nn.Module):
    """Pre-norm transformer layer; attention logits get an additive bias from pairwise formula differences (MIST)."""

    def __init__(self, d, h, p):
        super().__init__()
        self.h, self.n1, self.n2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, bias):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


class MISTEncoder(nn.Module):
    """Formula-transformer spectrum encoder in the spirit of MIST (Goldman et al. 2023).

    Tokens: one precursor token (full formula, metadata) + one token per peak (fragment subformula f, neutral loss
    F - f, Fourier m/z features, intensity, annotation type). Output: spectrum embedding + fingerprint logits."""

    def __init__(self, d, h, L, p, n_adducts, fp_bits, meta_dim=5):
        super().__init__()
        self.h, self.L = h, L
        tok_dim = 2 * N_EL + 32 + 3 + 5 + 1
        self.tok = nn.Sequential(nn.Linear(tok_dim, d), nn.GELU(), nn.Linear(d, d))
        self.meta = nn.Linear(meta_dim, d)
        self.adduct = nn.Embedding(n_adducts, d)
        self.pair = nn.Sequential(nn.Linear(N_EL + 1, 64), nn.GELU(), nn.Linear(64, h * L))
        self.layers = nn.ModuleList(PairBiasLayer(d, h, p) for _ in range(L))
        self.norm = nn.LayerNorm(d)
        self.fp_head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, fp_bits))

    def forward(self, s):
        sub, mz, it, typ, err, npk, fvec = s["sub"], s["mz"], s["int"], s["typ"], s["err"], s["npk"], s["fvec"]
        B, dev = sub.shape[0], sub.device
        sub_all = torch.cat([fvec[:, None], sub], 1)                                        # [B, N, 13]
        ann = torch.cat([torch.ones(B, 1, device=dev), (typ > 0).float()], 1)
        loss = (fvec[:, None] - sub_all).clamp(min=0) * ann[..., None]
        mz_all = torch.cat([s["prec"][:, None], mz], 1)
        typ_all = torch.cat([torch.full((B, 1), 4, device=dev, dtype=typ.dtype), typ], 1)
        feats = torch.cat([torch.log1p(sub_all) / 3, torch.log1p(loss) / 3, fourier(mz_all),
                           torch.sqrt(torch.cat([torch.ones(B, 1, device=dev), it], 1))[..., None], ann[..., None],
                           torch.cat([torch.zeros(B, 1, device=dev), err / 20], 1)[..., None],
                           F.one_hot(typ_all.long(), 5).float(), (typ_all == 4).float()[..., None]], -1)
        x = self.tok(feats)
        ctx = (self.meta(s["meta"]) + self.adduct(s["adduct"])).to(x.dtype)
        x = torch.cat([x[:, :1] + ctx[:, None], x[:, 1:]], 1)                              # metadata enters via the precursor token
        N = x.shape[1]
        valid = torch.arange(N, device=dev)[None] <= npk[:, None]                          # precursor + real peaks
        diff = sub_all[:, :, None] - sub_all[:, None, :]
        both = (ann[:, :, None] * ann[:, None, :])[..., None]
        pb = self.pair(torch.cat([torch.sign(diff) * torch.log1p(diff.abs()) * both, both], -1))  # [B,N,N,h*L]
        pb = pb.view(B, N, N, self.L, self.h).permute(3, 0, 4, 1, 2)
        mask = torch.zeros(B, 1, 1, N, device=dev).masked_fill(~valid[:, None, None], float("-inf"))
        for l, layer in enumerate(self.layers):
            x = layer(x, pb[l] + mask)
        z = self.norm(x[:, 0])
        return z, self.fp_head(z)


class GINELayer(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.bond = nn.ModuleList(nn.Embedding(v, d) for v in BOND_VOCAB)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.eps, self.norm, self.drop = nn.Parameter(torch.zeros(1)), nn.LayerNorm(d), nn.Dropout(p)

    def forward(self, h, src, dst, battr):
        e = sum(emb(battr[:, i]) for i, emb in enumerate(self.bond))
        agg = torch.zeros_like(h).index_add_(0, dst, F.relu(h[src] + e))
        return self.norm(h + self.drop(self.mlp((1 + self.eps) * h + agg)))


class GraphEncoder(nn.Module):
    def __init__(self, d, L, p):
        super().__init__()
        self.atom = nn.ModuleList(nn.Embedding(v, d) for v in ATOM_VOCAB)
        self.layers = nn.ModuleList(GINELayer(d, p) for _ in range(L))
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, g):
        h = sum(emb(g["x"][:, i]) for i, emb in enumerate(self.atom))
        for layer in self.layers:
            h = layer(h, g["src"], g["dst"], g["battr"])
        n = g["n_graphs"]
        cnt = torch.zeros(n, device=h.device).index_add_(0, g["batch"], torch.ones_like(g["batch"], dtype=h.dtype)).clamp(min=1)
        mean = torch.zeros(n, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, g["batch"], h) / cnt[:, None]
        mx = torch.full((n, h.shape[1]), -1e4, device=h.device, dtype=h.dtype).scatter_reduce(
            0, g["batch"][:, None].expand_as(h), h, "amax", include_self=True)
        return self.out(torch.cat([mean, mx], 1))


class DualReranker(nn.Module):
    """Scores (spectrum, candidate) pairs. use_graph=False gives the no-GNN ablation (everything else identical)."""
    N_SCALARS = 4   # ppm error, explained intensity, annotated-peak fraction, fingerprint log-likelihood

    def __init__(self, use_graph=True, *, n_adducts, d=256, heads=8, spec_layers=3, gnn_layers=4, dropout=0.1,
                 fp_bits=1024, **_):
        super().__init__()
        self.use_graph = use_graph
        self.spec = MISTEncoder(d, heads, spec_layers, dropout, n_adducts, fp_bits)
        self.ps = nn.Linear(d, d)
        if use_graph:
            self.graph = GraphEncoder(d, gnn_layers, dropout)
            self.pg = nn.Linear(d, d)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        in_dim = (2 * d + 1 if use_graph else d) + self.N_SCALARS
        self.head = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.mass_prior = nn.Parameter(torch.tensor([0.0, -2.0]))   # score += a*ppm + b*ppm^2 (Gaussian mass error)

    def encode_spectra(self, s, chunk=96):
        if self.training or s["sub"].shape[0] <= chunk:
            return self.spec(s)
        outs = [self.spec({k: v[i:i + chunk] for k, v in s.items()}) for i in range(0, s["sub"].shape[0], chunk)]
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])

    def forward(self, s, seq_of_cand, cand_fp, scal, g=None, zg=None):
        zs, fp_logits = self.encode_spectra(s)
        l = fp_logits[seq_of_cand].float()
        fp_ll = (cand_fp * l - F.softplus(l)).mean(1, keepdim=True)          # log p(candidate fingerprint | spectrum, formula)
        a = self.ps(zs)[seq_of_cand]
        feats = [torch.cat([scal, fp_ll * 10], 1)]
        if self.use_graph:
            b = self.pg(self.graph(g) if zg is None else zg)
            cos = F.cosine_similarity(a, b, dim=1)[:, None] * self.logit_scale.exp()
            feats = [a * b, (a - b).abs(), cos] + feats
        else:
            feats = [a] + feats
        ppm = scal[:, 0]
        prior = self.mass_prior[0] * ppm + self.mass_prior[1] * ppm ** 2
        return self.head(torch.cat(feats, 1)).squeeze(1) + prior, fp_logits, fp_ll.squeeze(1)


# --------------------------------------------------------------------------------------------- scoring
class Scorer:
    """Turns (spectrum, candidate list) pairs into model inputs, with a cache of peak annotations."""

    def __init__(self, cat, sp, ppm=10.0, annot_ppm=20.0, annot_abs=0.002):
        self.cat, self.sp, self.ppm, self.annot = cat, sp, ppm, (annot_ppm, annot_abs)
        self.cache = {}

    def build_pairs(self, pairs, cache=False):
        """Annotate each (spectrum q, formula id f) pair; grouped by formula so each enumeration is computed once."""
        pairs = sorted(set(pairs), key=lambda x: (x[1], x[0]))
        idx = {p: i for i, p in enumerate(pairs)}
        P = self.sp.MZ.shape[1]
        SUB = np.zeros((len(pairs), P, N_EL), np.uint8); ERR = np.zeros((len(pairs), P), np.int8); TYP = np.zeros((len(pairs), P), np.int8)
        for i, (q, f) in enumerate(pairs):
            hit = self.cache.get((q, f))
            if hit is None:
                n = self.sp.NPK[q]
                hit = annotate(self.sp.MZ[q, :n].astype(float), self.sp.SHIFTS[q], self.cat.FVEC[f], *self.annot)
                if cache:
                    self.cache[(q, f)] = hit
            n = len(hit[2])
            SUB[i, :n], ERR[i, :n], TYP[i, :n] = hit
        return idx, SUB, ERR, TYP

    def assemble(self, qs, cand_lists, pair_arrays):
        """Flatten (query, candidate list) into unique (query, formula) sequences + candidate rows."""
        cat, sp, dev = self.cat, self.sp, self.cat.device
        seq_key, seq_of_cand, flat_q, flat_c = {}, [], [], []
        for q, cl in zip(qs, cand_lists):
            for c in cl:
                key = (q, cat.fid[c])
                if key not in seq_key:
                    seq_key[key] = len(seq_key)
                seq_of_cand.append(seq_key[key]); flat_q.append(q); flat_c.append(c)
        keys = list(seq_key)
        idx, SUB, ERR, TYP = pair_arrays
        sq = np.array([k[0] for k in keys]); sf = np.array([k[1] for k in keys])
        pid = np.array([idx[k] for k in keys])
        t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)
        s = dict(sub=t(SUB[pid]), err=t(ERR[pid]), typ=t(TYP[pid], torch.long), mz=t(sp.MZ[sq]), int=t(sp.INT[sq]),
                 npk=t(sp.NPK[sq], torch.long), fvec=t(cat.FVEC[sf]), prec=t(sp.prec[sq]), meta=t(sp.META[sq]),
                 adduct=t(sp.ADDUCT_ID[sq], torch.long))
        soc = torch.tensor(seq_of_cand, device=dev)
        flat_q, flat_c = np.array(flat_q), np.array(flat_c)
        nm = sp.neutral[flat_q]
        ppm = np.abs(cat.mass[flat_c] - nm) / nm * 1e6 / self.ppm
        ann = (s["typ"] > 0).float()
        expl = (s["int"] * ann).sum(1) / s["int"].sum(1).clamp(min=1e-9)
        frac = ann.sum(1) / (s["int"] > 0).float().sum(1).clamp(min=1)
        scal = torch.cat([torch.from_numpy(np.clip(ppm, 0, 3)).float().to(dev)[:, None],
                          expl[soc][:, None], frac[soc][:, None]], 1)
        return s, soc, flat_c, scal, keys

    @torch.no_grad()
    def embed_catalog(self, model, bs=1024, ids=None):
        """Graph embeddings of the catalog, kept in CPU memory (725k x 256 floats would not fit beside training on a
        4 GB GPU). With `ids`, only those rows are computed and the rest stay zero."""
        model.eval()
        if self.cat.device.type == "cuda":
            torch.cuda.empty_cache()
        ids = np.arange(self.cat.N) if ids is None else np.asarray(ids)
        out = None
        for i in range(0, len(ids), bs):
            z = model.graph(self.cat.graph_batch(ids[i:i + bs])).float().cpu()
            if out is None:
                out = torch.zeros(self.cat.N, z.shape[1])
            out[torch.as_tensor(ids[i:i + bs])] = z
        return out

    @torch.no_grad()
    def score_pools(self, model, items, zg_all=None, chunk=24):
        """items: list of (q, candidate index array) -> per item {model.name, 'MIST FP', 'Mass only': scores}."""
        model.eval()
        out = []
        for i in range(0, len(items), chunk):
            part = items[i:i + chunk]
            pa = self.build_pairs([(q, self.cat.fid[c]) for q, p in part for c in p], cache=True)
            s, soc, flat_c, scal, _ = self.assemble([q for q, _ in part], [p for _, p in part], pa)
            zg = zg_all[torch.as_tensor(flat_c)].to(self.cat.device) if model.use_graph else None
            score, _, fp_ll = model(s, soc, self.cat.unpack_fp(flat_c), scal, zg=zg)
            score, fp_ll = score.float().cpu().numpy(), fp_ll.float().cpu().numpy()
            off = 0
            for q, p in part:
                n = len(p)
                out.append({model.name: score[off:off + n], "MIST FP": fp_ll[off:off + n],
                            "Mass only": -np.round(np.abs(self.cat.mass[p] - self.sp.neutral[q]), 6)})
                off += n
        return out

    def own_pool(self, q):
        nm = self.sp.neutral[q]
        lo, hi = self.cat.window(nm, self.ppm) if np.isfinite(nm) else (0, 0)
        return np.arange(lo, hi)

    def molecule_pool(self, qs, fill=0):
        """Pool around the median neutral mass of a molecule's spectra (optionally padded with nearest masses)."""
        nm = np.nanmedian(self.sp.neutral[qs])
        lo, hi = self.cat.window(nm, self.ppm)
        if hi - lo < fill:
            near = np.argsort(np.abs(self.cat.mass[max(0, lo - fill):hi + fill] - nm), kind="stable")[:fill] + max(0, lo - fill)
            return np.unique(np.concatenate([np.arange(lo, hi), near]))
        return np.arange(lo, hi)


def aggregate(score_list):
    """Combine a molecule's spectra: mean of per-spectrum log-softmax over the shared candidate pool."""
    ls = [sc - np.logaddexp.reduce(sc) for sc in score_list]
    return np.round(np.mean(ls, 0), 9)


def tie_metrics(scores, true_pos):
    """Top-1, Top-5, MRR@25 and expected rank, averaging over the possible positions of tied scores."""
    s_true = scores[true_pos]
    better = int((scores > s_true).sum()); ties = int((scores == s_true).sum())
    ranks = np.arange(better + 1, better + ties + 1)
    return dict(top1=float(np.mean(ranks <= 1)), top5=float(np.mean(ranks <= 5)),
                mrr25=float(np.mean(np.where(ranks <= 25, 1 / ranks, 0))), rank=float(ranks.mean()))
