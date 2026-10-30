import os
import sys
import json
import numpy as np

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, save_json
from src.data_loader import prepare_and_save_data
from src.models.tree_baseline import SingleTreeBaseline
from src.models.bagging import CustomBaggingClassifier
from src.models.boosting import SAMMEAdaBoost
from src.metrics import compute_accuracy, compute_bias_variance

def run_q3():
    print("Executing Q3: Bias and Variance as Measured Quantities...")
    config = load_config()
    data = prepare_and_save_data(config)

    X_train_orig, y_train_orig = data["X_train"], data["y_train"]
    X_test_clean, y_test = data["X_test_clean"], data["y_test"]
    X_test_masked = data["X_test_masked"]

    rng = np.random.default_rng(config["seed"])
    R = config["q3"]["num_resamples"]  # 30 repetitions

    # Load frozen optimal hyperparameters from Q1 and Q2 logs if available
    frozen_tree_d, frozen_tree_m = 8, 1
    best_boost_d, best_boost_T = 2, 50

    try:
        with open("logs/q1_results.json") as f:
            q1_log = json.load(f)
            frozen_tree_d = q1_log["frozen_tree_params"]["max_depth"]
            frozen_tree_m = q1_log["frozen_tree_params"]["min_samples_leaf"]
    except Exception:
        pass

    try:
        with open("logs/q2_results.json") as f:
            q2_log = json.load(f)
            best_boost_d = q2_log["selected_setting"]["depth"]
            best_boost_T = q2_log["selected_setting"]["rounds"]
    except Exception:
        pass

    print(f"\n[Selected Hyperparameters for Q3 Evaluation]:")
    print(f"  - Single Tree & Bagging: max_depth={frozen_tree_d}, min_samples_leaf={frozen_tree_m}")
    print(f"  - Boosting: weak_depth={best_boost_d}, rounds={best_boost_T}")

    # -------------------------------------------------------------------------
    # Q3(a): 30 Resampled Fits for Selected Single Tree, Bagging, and Boosting
    # -------------------------------------------------------------------------
    print(f"\n--- Part (a): Fitting Models Across {R} Resampled Sets ---")
    
    tree_preds, bag_preds, boost_preds = [], [], []
    tree_accs, bag_accs, boost_accs = [], [], []

    for r in range(R):
        # Sampling with replacement, within each identity, 6 faces from original 6 training faces
        resampled_indices = []
        for identity in range(40):
            id_idx = np.where(y_train_orig == identity)[0]
            sampled_idx = rng.choice(id_idx, size=6, replace=True)
            resampled_indices.extend(sampled_idx)

        X_resampled = X_train_orig[resampled_indices]
        y_resampled = y_train_orig[resampled_indices]

        # 1. Single Tree Baseline
        tree = SingleTreeBaseline(max_depth=frozen_tree_d, min_samples_leaf=frozen_tree_m, random_state=rng.integers(0, 100000))
        tree.fit(X_resampled, y_resampled)
        tree_preds.append(tree.predict_proba(X_test_clean))
        tree_accs.append(compute_accuracy(y_test, tree.predict(X_test_clean)))

        # 2. Bagging Model
        bagger = CustomBaggingClassifier(n_estimators=100, max_depth=frozen_tree_d, min_samples_leaf=frozen_tree_m, random_state=rng.integers(0, 100000))
        bagger.fit(X_resampled, y_resampled)
        bag_preds.append(bagger.predict_proba(X_test_clean))
        bag_accs.append(compute_accuracy(y_test, bagger.predict(X_test_clean)))

        # 3. Boosting Model (using normalized weighted vote vector)
        booster = SAMMEAdaBoost(n_estimators=best_boost_T, max_depth=best_boost_d, n_classes=40, random_state=rng.integers(0, 100000))
        booster.fit(X_resampled, y_resampled)
        boost_preds.append(booster.predict_proba(X_test_clean))
        boost_accs.append(compute_accuracy(y_test, booster.predict(X_test_clean)))

        if (r + 1) % 10 == 0 or r == 0:
            print(f"  Completed Resample Repetition {r+1}/{R}")

    # Calculate Bias^2 and Variance metrics
    tree_b2, tree_var = compute_bias_variance(tree_preds, y_test)
    bag_b2, bag_var = compute_bias_variance(bag_preds, y_test)
    boost_b2, boost_var = compute_bias_variance(boost_preds, y_test)

    print("\n--- Part (a) Measured Results ---")
    print(f"Single Tree -> Bias^2: {tree_b2:.6f} | Variance: {tree_var:.6f} | Acc: {np.mean(tree_accs):.4f} +/- {np.std(tree_accs):.4f}")
    print(f"Bagging     -> Bias^2: {bag_b2:.6f} | Variance: {bag_var:.6f} | Acc: {np.mean(bag_accs):.4f} +/- {np.std(bag_accs):.4f}")
    print(f"Boosting    -> Bias^2: {boost_b2:.6f} | Variance: {boost_var:.6f} | Acc: {np.mean(boost_accs):.4f} +/- {np.std(boost_accs):.4f}")

    # -------------------------------------------------------------------------
    # Q3(b): Depth-wise Bias-Variance Analysis (Shallow, Selected, Deep)
    # -------------------------------------------------------------------------
    print("\n--- Part (b): Depth-wise Bias/Variance Quantification ---")
    depths_to_test = [2, frozen_tree_d if frozen_tree_d is not None else 8, 16]
    depth_analysis_results = []

    for d in depths_to_test:
        d_label = "None/16" if d == 16 else str(d)
        st_preds, bg_preds = [], []
        st_accs, bg_accs = [], []

        for r in range(R):
            resampled_indices = []
            for identity in range(40):
                id_idx = np.where(y_train_orig == identity)[0]
                sampled_idx = rng.choice(id_idx, size=6, replace=True)
                resampled_indices.extend(sampled_idx)

            X_resampled = X_train_orig[resampled_indices]
            y_resampled = y_train_orig[resampled_indices]

            # Single Tree at depth d
            t = SingleTreeBaseline(max_depth=d, min_samples_leaf=1, random_state=rng.integers(0, 100000))
            t.fit(X_resampled, y_resampled)
            st_preds.append(t.predict_proba(X_test_clean))
            st_accs.append(compute_accuracy(y_test, t.predict(X_test_clean)))

            # 100-Tree Bagging Ensemble at depth d
            b = CustomBaggingClassifier(n_estimators=100, max_depth=d, min_samples_leaf=1, random_state=rng.integers(0, 100000))
            b.fit(X_resampled, y_resampled)
            bg_preds.append(b.predict_proba(X_test_clean))
            bg_accs.append(compute_accuracy(y_test, b.predict(X_test_clean)))

        st_b2, st_v = compute_bias_variance(st_preds, y_test)
        bg_b2, bg_v = compute_bias_variance(bg_preds, y_test)
        variance_reduction = float(st_v - bg_v)

        depth_analysis_results.append({
            "depth": d,
            "single_tree": {
                "bias_squared": float(st_b2),
                "variance": float(st_v),
                "test_acc_mean": float(np.mean(st_accs)),
                "test_acc_std": float(np.std(st_accs))
            },
            "bagging_100": {
                "bias_squared": float(bg_b2),
                "variance": float(bg_v),
                "test_acc_mean": float(np.mean(bg_accs)),
                "test_acc_std": float(np.std(bg_accs))
            },
            "variance_removed_by_bagging": variance_reduction
        })
        print(f"Depth {d_label:>4} -> Tree Var: {st_v:.6f} | Bag Var: {bg_v:.6f} | Var Reduced: {variance_reduction:.6f}")

    # -------------------------------------------------------------------------
    # Q3(d): Consolidated Model Summary Table Data
    # -------------------------------------------------------------------------
    print("\n--- Part (d): Building Consolidated Comparison Summary ---")
    
    # Fit models on the full original training set for clean and masked test evaluation
    final_tree = SingleTreeBaseline(max_depth=frozen_tree_d, min_samples_leaf=frozen_tree_m, random_state=config["seed"]).fit(X_train_orig, y_train_orig)
    final_bagger = CustomBaggingClassifier(n_estimators=100, max_depth=frozen_tree_d, min_samples_leaf=frozen_tree_m, random_state=config["seed"]).fit(X_train_orig, y_train_orig)
    final_booster = SAMMEAdaBoost(n_estimators=best_boost_T, max_depth=best_boost_d, n_classes=40, random_state=config["seed"]).fit(X_train_orig, y_train_orig)

    summary_table = {
        "Single Tree": {
            "clean_test_acc": float(compute_accuracy(y_test, final_tree.predict(X_test_clean))),
            "masked_test_acc": float(compute_accuracy(y_test, final_tree.predict(X_test_masked))),
            "bias_squared": float(tree_b2),
            "variance": float(tree_var)
        },
        "Bagging": {
            "clean_test_acc": float(compute_accuracy(y_test, final_bagger.predict(X_test_clean))),
            "masked_test_acc": float(compute_accuracy(y_test, final_bagger.predict(X_test_masked))),
            "bias_squared": float(bag_b2),
            "variance": float(bag_var)
        },
        "Boosting": {
            "clean_test_acc": float(compute_accuracy(y_test, final_booster.predict(X_test_clean))),
            "masked_test_acc": float(compute_accuracy(y_test, final_booster.predict(X_test_masked))),
            "bias_squared": float(boost_b2),
            "variance": float(boost_var)
        }
    }

    # Prepare JSON Output
    output = {
        "part_a_30_resamples": {
            "single_tree": {
                "bias_squared": float(tree_b2),
                "variance": float(tree_var),
                "test_acc_mean": float(np.mean(tree_accs)),
                "test_acc_std": float(np.std(tree_accs))
            },
            "bagging": {
                "bias_squared": float(bag_b2),
                "variance": float(bag_var),
                "test_acc_mean": float(np.mean(bag_accs)),
                "test_acc_std": float(np.std(bag_accs))
            },
            "boosting": {
                "bias_squared": float(boost_b2),
                "variance": float(boost_var),
                "test_acc_mean": float(np.mean(boost_accs)),
                "test_acc_std": float(np.std(boost_accs))
            }
        },
        "part_b_depth_analysis": depth_analysis_results,
        "part_d_summary_table": summary_table
    }

    os.makedirs("logs", exist_ok=True)
    save_json(output, "logs/q3_results.json")
    print("\nQ3 completed successfully. Results saved to logs/q3_results.json")

if __name__ == "__main__":
    run_q3()