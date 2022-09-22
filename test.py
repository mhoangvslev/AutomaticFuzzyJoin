from autofj.datasets import load_data
from autofj import AutoFJ

import pandas as pd
import numpy as np
import os
import sys
import json

# def tracefunc(frame, event, arg, indent=[0]):
#    if event == "call":
#        indent[0] += 2
#        print("-" * indent[0] + "> call function", frame.f_code.co_name, "File: ", os.path.realpath(frame.f_code.co_filename))
#    elif event == "return":
#        print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
#        indent[0] -= 2
#    return tracefunc  
# sys.setprofile(tracefunc)

# Evaluate
def evaluate(y_true: pd.DataFrame, y_pred: pd.DataFrame, **kwargs):

    y_pred = y_pred[y_pred["id_l"].isin(y_true["id_l"])]

    gt_joins = y_true.astype(str)[["id_l", "id_r"]].values
    pred_joins = y_pred.astype(str)[["id_l", "id_r"]].values

    #pred_set = {tuple(sorted(j)) for j in pred_joins}
    #gt_set = {tuple(sorted(j)) for j in gt_joins}

    pred_set = {(l, r) for l, r in pred_joins}
    gt_set = {(l, r) for l, r in gt_joins}

    # TP: When the prediction is in ground truth
    tp = pred_set.intersection(gt_set)
    fp = pred_set.difference(tp)
    fn = gt_set.difference(tp)
    
    try: precision = len(tp) / len(pred_set)
    except: precision = np.nan

    try: recall = len(tp) / len(gt_set)
    except: recall = np.nan

    try:
        f_coef = kwargs.get('f_coef')
        f_coef = 1 if f_coef is None else f_coef
        fscore = (f_coef + 1) * precision * recall / (precision + recall)
    except:
        fscore = np.nan
    
    test_results = {'precision': precision, 'recall': recall, f'f{f_coef}-score': fscore}

    if kwargs.get("verbose"):
        return test_results, tp, fp, fn
    return test_results

# Main
if __name__ == "__main__":
    args = sys.argv[1:]
    dataset = args[0]
    target_precision = float(args[1])
    outdir = os.path.join("exp", dataset)
    os.makedirs(outdir, exist_ok=True)

    # Train model
    left_table, right_table, gt_table = load_data(dataset)
    left_table = left_table.astype(str)

    fj = AutoFJ(precision_target=target_precision)
    result = fj.join(left_table, right_table, "id")

    test_results, tp, fp, fn = evaluate(gt_table, result, verbose=True)

    # Export results
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(test_results, f)
    
    print(test_results)

    tp = (
        pd.DataFrame(tp, columns=['id_l', 'id_r'])
            .merge(left_table.rename(columns={'id': 'id_l'}), on="id_l", suffixes=("_l", "_r"))
            .merge(right_table.rename(columns={'id': 'id_r'}), on="id_r", suffixes=("_l", "_r"))
    )
        
    fp = (
        pd.DataFrame(fp, columns=['id_l', 'id_r'])
            .merge(left_table.rename(columns={'id': 'id_l'}), on="id_l", suffixes=("_l", "_r"))
            .merge(right_table.rename(columns={'id': 'id_r'}), on="id_r", suffixes=("_l", "_r"))
    )

    fn = (
        pd.DataFrame(fn, columns=['id_l', 'id_r'])
            .merge(left_table.rename(columns={'id': 'id_l'}), on="id_l", suffixes=("_l", "_r"))
            .merge(right_table.rename(columns={'id': 'id_r'}), on="id_r", suffixes=("_l", "_r"))
    )

    result.to_csv(os.path.join(outdir, "pred.csv"), index=False)
    tp.to_csv(os.path.join(outdir, "pred_tp.csv"), index=False)
    fp.to_csv(os.path.join(outdir, "pred_fp.csv"), index=False)
    fn.to_csv(os.path.join(outdir, "pred_fn.csv"), index=False)

    learned_col_weights = pd.DataFrame({ k: [v] for k, v in fj.selected_column_weights.items()})
    learned_col_weights.to_csv(os.path.join(outdir, "learned_col_weight.csv"), index=False)
    learned_join_conf = pd.DataFrame(fj.selected_join_configs, columns=["config", "threshhold"])
    learned_join_conf.to_csv(os.path.join(outdir, "learned_join_conf.csv"), index=False)

