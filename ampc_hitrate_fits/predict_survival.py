import json
import numpy as np
from model import HitRateModel
from combine_all_data_pprop import (
    make_pprop_interpolator,
    load_13b_anion_score_pprop_dict,
)


PPROP_INTER = make_pprop_interpolator(load_13b_anion_score_pprop_dict())


def predict_survival(docking_scores, pki_min=3, pki_max=9, pki_step=0.01):
    pprops = PPROP_INTER(docking_scores)
    with open("fitted_params.json", "r") as f:
        params = json.load(f)["ampc"]
    model = HitRateModel(params)
    pki_grid = np.arange(pki_min, pki_max, pki_step)
    survival = np.zeros((len(pki_grid), 2))
    survival[:, 0] = pki_grid
    for j, y in enumerate(pki_grid):
        defs = np.full_like(pprops, y)
        survival[j, 1] = model.get_pprop_hr(pprops, defs).mean()
    return survival


if __name__ == "__main__":
    print(predict_survival([-85, -86, -70, -68]))
