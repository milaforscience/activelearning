import numpy as np
from scipy.interpolate import interp1d


def make_pprop_interpolator(score_pprop_dict, kind="linear", extrapolate=False):
    """
    Build an interpolation function from a {score_str: pprop} dict.

    Parameters
    ----------
    score_pprop_dict : dict
        Keys are score strings (e.g. '-7.25'), values are pProp floats.
    kind : str
        Interpolation kind passed to scipy.interpolate.interp1d
        ('linear', 'cubic', 'quadratic', 'nearest', ...).
    extrapolate : bool
        If False, scores outside the known range raise ValueError.
        If True, values are extrapolated (use with caution).

    Returns
    -------
    f : callable
        f(score) -> pprop. Accepts a scalar or array-like.
    """
    # Convert keys to floats and sort by score
    scores = np.array([float(k) for k in score_pprop_dict.keys()])
    pprops = np.array(list(score_pprop_dict.values()))
    order = np.argsort(scores)
    scores = scores[order]
    pprops = pprops[order]

    fill_value = "extrapolate" if extrapolate else np.nan
    bounds_error = not extrapolate

    interp = interp1d(
        scores,
        pprops,
        kind=kind,
        bounds_error=bounds_error,
        fill_value=fill_value,
        assume_sorted=True,
    )

    s_min, s_max = scores[0], scores[-1]

    def pprop_fn(score):
        score_arr = np.asarray(score, dtype=float)
        result = interp(score_arr)
        return float(result) if result.ndim == 0 else result

    pprop_fn.score_min = s_min
    pprop_fn.score_max = s_max
    return pprop_fn


def combine_chemstep_data():
    interp = make_pprop_interpolator(load_13b_anion_score_pprop_dict())
    fout = open("dataframes/chemstep_hits_pprop_pki.df", "w")
    allout = open("dataframes/all_pprop_pki.df", "w")
    fout.write("pprop pki\n")
    allout.write("pprop pki\n")
    with open("data/novel_tested_compounds_zids.df") as f:
        lines = f.readlines()
    for line in lines[1:]:
        ll = line.split()
        try:
            ki = float(ll[-3])
        except ValueError:
            ki = 10**6
        pki = -1 * np.log10(ki / 10**6)
        pprop = interp(float(ll[-2]))
        fout.write(f"{pprop} {pki}\n")
        allout.write(f"{pprop} {pki}\n")
    fout.close()
    for fn in ["data/ampc_pprops.df", "data/ampc_yujin_pprops.df"]:
        with open("data/ampc_pprops.df") as f:
            lines = f.readlines()
        for line in lines[1:]:
            ll = line.split()
            try:
                ki = float(ll[0])
            except ValueError:
                ki = 10**6
            pki = -1 * np.log10(ki / 10**6)
            pprop = float(ll[-1])
            allout.write(f"{pprop} {pki}\n")
    allout.close()


def load_13b_anion_score_pprop_dict():
    score_pprop_dict = {}
    with open("data/full_scores.df") as f:
        lines = f.readlines()
    for line in lines[1:]:
        ll = line.split()
        score = f"{np.round(float(ll[0]), decimals=2):.2f}"
        pprop = float(ll[-1])
        score_pprop_dict[score] = pprop
    return score_pprop_dict


if __name__ == "__main__":
    combine_chemstep_data()
