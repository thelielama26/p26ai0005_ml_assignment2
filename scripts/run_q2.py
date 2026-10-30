import os
import sys
import time
import numpy as np
import json

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, save_json
from src.data_loader import prepare_and_save_data
from src.models.tree_baseline import SingleTreeBaseline
from src.models.bagging import CustomBaggingClassifier
from src.models.boosting import SAMMEAdaBoost
from src.metrics import compute_accuracy, compute_confidence

def run_q2():
    print("Executing Q2: Multiclass Boosting from Scratch (SAMME AdaBoost)...")
    config = load_config()
    data = prepare_and_save_data(config)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test_clean, y_test = data["X_test_clean"], data["y_test"]
    X_test_masked = data["X_test_masked"]

    # -------------------------------------------------------------------------
    # Q2(a): 3-Class Toy Example Verification (<= 10 samples)
    # -------------------------------------------------------------------------
    print("\n--- Part (a): 3-Class Toy Example Verification (Round 1 Logging) ---")
    np.random.seed(42)
    # 6 samples, 2 features, 3 classes (K=3)
    X_toy = np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [0.8, 0.9],
        [0.9, 0.8],
        [0.5, 0.5],
        [0.6, 0.4]
    ])
    y_toy = np.array([0, 0, 1, 1, 2, 2])

    toy_booster = SAMMEAdaBoost(n_estimators=1, max_depth=1, n_classes=3, random_state=42)
    toy_log = toy_booster.fit(X_toy, y_toy, verbose_log=True)

    print("Toy Example Initial Weights:", np.round(np.ones(6) / 6.0, 4))
    print("Toy Example Round 1 Predictions:", toy_log[0]["predictions"])
    print(f"Toy Example Round 1 Weighted Error: {toy_log[0]['weighted_error']:.4f}")
    print(f"Toy Example Round 1 Alpha_t: {toy_log[0]['alpha']:.4f}")
    print("Toy Example Updated Normalized Weights:", np.round(toy_log[0]["sample_weights"], 4))

    # -------------------------------------------------------------------------
    # Q2(b): Hyperparameter Grid Search on Olivetti Faces
    # -------------------------------------------------------------------------
    print("\n--- Part (b): Grid Search over Weak Learner Depths and Boosting Rounds ---")
    grid_search_results = []
    best_val_acc = -1.0
    best_params = None

    for d in config["q2"]["weak_depths"]:
        for T in config["q2"]["rounds"]:
            model = SAMMEAdaBoost(n_estimators=T, max_depth=d, n_classes=40, random_state=config["seed"])
            model.fit(X_train, y_train)

            tr_acc = compute_accuracy(y_train, model.predict(X_train))
            val_acc = compute_accuracy(y_val, model.predict(X_val))
            te_acc = compute_accuracy(y_test, model.predict(X_test_clean))

            res = {
                "weak_depth": d,
                "rounds": T,
                "train_acc": float(tr_acc),
                "val_acc": float(val_acc),
                "test_acc": float(te_acc)
            }
            grid_search_results.append(res)

            print(f"Depth: {d:>2} | Rounds: {T:>3} -> Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {te_acc:.4f}")

            # Select model using clean validation accuracy only
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (d, T)

    best_d, best_T = best_params
    print(f"\n[Selected Boosting Hyperparameters]: depth={best_d}, rounds={best_T} (Val Acc: {best_val_acc:.4f})")

    # Fit selected model with full round-by-round logging
    selected_booster = SAMMEAdaBoost(n_estimators=best_T, max_depth=best_d, n_classes=40, random_state=config["seed"])
    rounds_history = selected_booster.fit(X_train, y_train, verbose_log=True)

    # -------------------------------------------------------------------------
    # Q2(c): Boosting vs. Bagging Comparison & Hard Sample Inspection
    # -------------------------------------------------------------------------
    print("\n--- Part (c): Boosting vs. Bagging Comparison (25, 50, 100 Learners) ---")
    
    # Load frozen tree parameters from Q1 or fallback
    try:
        with open("logs/q1_results.json") as f:
            q1_log = json.load(f)
            frozen_d = q1_log["frozen_tree_params"]["max_depth"]
            frozen_m = q1_log["frozen_tree_params"]["min_samples_leaf"]
    except Exception:
        frozen_d, frozen_m = 8, 1

    comparison_results = []

    for B in [25, 50, 100]:
        # Evaluate Bagging
        t0 = time.time()
        bagger = CustomBaggingClassifier(n_estimators=B, max_depth=frozen_d, min_samples_leaf=frozen_m, random_state=config["seed"])
        bagger.fit(X_train, y_train)
        bag_time = time.time() - t0

        bag_tr = compute_accuracy(y_train, bagger.predict(X_train))
        bag_val = compute_accuracy(y_val, bagger.predict(X_val))
        bag_te = compute_accuracy(y_test, bagger.predict(X_test_clean))

        # Evaluate Boosting
        t0 = time.time()
        booster = SAMMEAdaBoost(n_estimators=B, max_depth=best_d, n_classes=40, random_state=config["seed"])
        booster.fit(X_train, y_train)
        boost_time = time.time() - t0

        boost_tr = compute_accuracy(y_train, booster.predict(X_train))
        boost_val = compute_accuracy(y_val, booster.predict(X_val))
        boost_te = compute_accuracy(y_test, booster.predict(X_test_clean))

        comparison_results.append({
            "learners": B,
            "bagging": {
                "train_acc": float(bag_tr),
                "val_acc": float(bag_val),
                "test_acc": float(bag_te),
                "train_time_sec": float(bag_time)
            },
            "boosting": {
                "train_acc": float(boost_tr),
                "val_acc": float(boost_val),
                "test_acc": float(boost_te),
                "train_time_sec": float(boost_time)
            }
        })
        print(f"Learners={B:>3} | Bagging Test Acc: {bag_te:.4f} ({bag_time:.2f}s) | Boosting Test Acc: {boost_te:.4f} ({boost_time:.2f}s)")

    # Hard sample inspection: find the training face with highest final weight
    final_weights = np.array(rounds_history[-1]["sample_weights"])
    heaviest_sample_idx = int(np.argmax(final_weights))
    heaviest_sample_weight = float(final_weights[heaviest_sample_idx])
    heaviest_sample_true_class = int(y_train[heaviest_sample_idx])

    # Extract prediction history across rounds for this sample
    prediction_history = []
    for step in rounds_history:
        preds = step["predictions"]
        prediction_history.append({
            "round": step["round"],
            "sample_weight": float(step["sample_weights"][heaviest_sample_idx]),
            "predicted_class": int(preds[heaviest_sample_idx]),
            "is_correct": bool(preds[heaviest_sample_idx] == heaviest_sample_true_class)
        })

    # -------------------------------------------------------------------------
    # Q2(d): Robustness Evaluation across Baseline, Bagging, and Boosting
    # -------------------------------------------------------------------------
    print("\n--- Part (d): Robustness Evaluation & Differential Response ---")
    
    tree_final = SingleTreeBaseline(max_depth=frozen_d, min_samples_leaf=frozen_m, random_state=config["seed"])
    tree_final.fit(X_train, y_train)
    
    bag_final = CustomBaggingClassifier(n_estimators=100, max_depth=frozen_d, min_samples_leaf=frozen_m, random_state=config["seed"])
    bag_final.fit(X_train, y_train)
    
    boost_final = SAMMEAdaBoost(n_estimators=best_T, max_depth=best_d, n_classes=40, random_state=config["seed"])
    boost_final.fit(X_train, y_train)

    robustness_metrics = {
        "tree": {
            "clean_acc": float(compute_accuracy(y_test, tree_final.predict(X_test_clean))),
            "clean_conf": float(compute_confidence(tree_final.predict_proba(X_test_clean))),
            "masked_acc": float(compute_accuracy(y_test, tree_final.predict(X_test_masked))),
            "masked_conf": float(compute_confidence(tree_final.predict_proba(X_test_masked)))
        },
        "bagging": {
            "clean_acc": float(compute_accuracy(y_test, bag_final.predict(X_test_clean))),
            "clean_conf": float(compute_confidence(bag_final.predict_proba(X_test_clean))),
            "masked_acc": float(compute_accuracy(y_test, bag_final.predict(X_test_masked))),
            "masked_conf": float(compute_confidence(bag_final.predict_proba(X_test_masked)))
        },
        "boosting": {
            "clean_acc": float(compute_accuracy(y_test, boost_final.predict(X_test_clean))),
            "clean_conf": float(compute_confidence(boost_final.predict_proba(X_test_clean))),
            "masked_acc": float(compute_accuracy(y_test, boost_final.predict(X_test_masked))),
            "masked_conf": float(compute_confidence(boost_final.predict_proba(X_test_masked)))
        }
    }

    # Find a face sample where bagging and boosting respond differently to occlusion
    bag_masked_preds = bag_final.predict(X_test_masked)
    boost_masked_preds = boost_final.predict(X_test_masked)
    
    diff_indices = np.where(bag_masked_preds != boost_masked_preds)[0]
    case_study = None
    if len(diff_indices) > 0:
        c_idx = int(diff_indices[0])
        case_study = {
            "test_sample_index": c_idx,
            "true_label": int(y_test[c_idx]),
            "bagging_masked_prediction": int(bag_masked_preds[c_idx]),
            "boosting_masked_prediction": int(boost_masked_preds[c_idx]),
            "bagging_votes": bag_final.predict_vote_counts(X_test_masked[c_idx:c_idx+1])[0].tolist(),
            "boosting_probabilities": boost_final.predict_proba(X_test_masked[c_idx:c_idx+1])[0].tolist()
        }

    # Prepare JSON Output
    output = {
        "toy_example_round1": toy_log[0],
        "grid_search": grid_search_results,
        "selected_setting": {
            "depth": best_d,
            "rounds": best_T
        },
        "selected_model_history": rounds_history,
        "bagging_vs_boosting_comparison": comparison_results,
        "heaviest_sample_inspection": {
            "sample_index": heaviest_sample_idx,
            "final_weight": heaviest_sample_weight,
            "true_class": heaviest_sample_true_class,
            "history": prediction_history
        },
        "robustness_summary": robustness_metrics,
        "differential_response_case_study": case_study
    }

    os.makedirs("logs", exist_ok=True)
    save_json(output, "logs/q2_results.json")
    print("\nQ2 completed successfully. Results saved to logs/q2_results.json")

if __name__ == "__main__":
    run_q2()