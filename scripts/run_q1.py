import os
import sys
import numpy as np

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, save_json
from src.data_loader import prepare_and_save_data
from src.models.tree_baseline import SingleTreeBaseline
from src.models.bagging import CustomBaggingClassifier
from src.metrics import compute_accuracy, compute_confidence

def run_q1():
    print("Executing Q1: Decision tree baseline, Bagging from scratch, and Robustness Experiment...")
    config = load_config()
    data = prepare_and_save_data(config)
    
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test_clean, y_test = data["X_test_clean"], data["y_test"]
    X_test_masked = data["X_test_masked"]

    # -------------------------------------------------------------------------
    # Q1(a): Single Tree Baseline Sweep
    # -------------------------------------------------------------------------
    print("\n--- Part (a): Decision Tree Baseline Grid Search ---")
    best_val_acc = -1.0
    best_params = None
    baseline_sweep_results = []
    
    for d in config["q1"]["max_depths"]:
        for m in config["q1"]["min_samples_leafs"]:
            tree = SingleTreeBaseline(max_depth=d, min_samples_leaf=m, random_state=config["seed"])
            tree.fit(X_train, y_train)
            
            tr_acc = compute_accuracy(y_train, tree.predict(X_train))
            val_acc = compute_accuracy(y_val, tree.predict(X_val))
            te_acc = compute_accuracy(y_test, tree.predict(X_test_clean))
            
            stats = tree.get_tree_stats(X_train)
            
            res = {
                "max_depth": d,
                "min_samples_leaf": m,
                "train_acc": float(tr_acc),
                "val_acc": float(val_acc),
                "test_acc": float(te_acc),
                **stats
            }
            baseline_sweep_results.append(res)
            
            print(f"Depth: {str(d):>4} | MinLeaf: {m} -> Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {te_acc:.4f} | Leaves: {stats['num_leaves']}")
            
            # Select strictly using validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (d, m)
                
    frozen_d, frozen_m = best_params
    print(f"\n[Selected Baseline Hyperparameters]: max_depth={frozen_d}, min_samples_leaf={frozen_m} (Val Acc: {best_val_acc:.4f})")

    # -------------------------------------------------------------------------
    # Q1(b) & Q1(c): Bagging Sweep & Analysis
    # -------------------------------------------------------------------------
    print("\n--- Parts (b) & (c): Custom Bagging Sweep ---")
    n_samples = X_train.shape[0]
    theoretical_expectation = n_samples * (1.0 - (1.0 - 1.0 / n_samples) ** n_samples)
    print(f"Theoretical expected unique samples in bootstrap size {n_samples}: {theoretical_expectation:.2f}")

    bagging_sweep_results = []
    best_bag_val_acc = -1.0
    selected_B = None

    for B in config["q1"]["ensemble_sizes"]:
        bagger = CustomBaggingClassifier(
            n_estimators=B, 
            max_depth=frozen_d, 
            min_samples_leaf=frozen_m, 
            random_state=config["seed"]
        )
        bagger.fit(X_train, y_train)
        
        tr_acc = compute_accuracy(y_train, bagger.predict(X_train))
        val_acc = compute_accuracy(y_val, bagger.predict(X_val))
        te_acc = compute_accuracy(y_test, bagger.predict(X_test_clean))
        
        # Coverage calculation
        unique_counts = [len(np.unique(idx)) for idx in bagger.bootstrap_indices]
        avg_unique = float(np.mean(unique_counts))
        
        oob_acc = bagger.compute_oob_accuracy(X_train, y_train)
        disagreement, margin = bagger.compute_metrics(X_val)
        
        bagging_sweep_results.append({
            "B": B,
            "train_acc": float(tr_acc),
            "val_acc": float(val_acc),
            "test_acc": float(te_acc),
            "avg_unique_bootstrap": avg_unique,
            "theoretical_unique_bootstrap": float(theoretical_expectation),
            "oob_acc": float(oob_acc),
            "disagreement": float(disagreement),
            "vote_margin": float(margin)
        })

        print(f"B={B:>3} -> Val Acc: {val_acc:.4f} | Test Acc: {te_acc:.4f} | OOB Acc: {oob_acc:.4f} | Disagreement: {disagreement:.4f} | Margin: {margin:.4f}")
        
        if val_acc > best_bag_val_acc:
            best_bag_val_acc = val_acc
            selected_B = B

    print(f"\n[Selected Ensemble Size]: B={selected_B} (Val Acc: {best_bag_val_acc:.4f})")

    # -------------------------------------------------------------------------
    # Q1(d): Robustness Experiment (Clean vs Masked) & Correction Analysis
    # -------------------------------------------------------------------------
    print("\n--- Part (d): Robustness Evaluation & Error Correction Analysis ---")
    
    # Train frozen single tree & final selected bagging model
    final_tree = SingleTreeBaseline(max_depth=frozen_d, min_samples_leaf=frozen_m, random_state=config["seed"])
    final_tree.fit(X_train, y_train)
    
    final_bagger = CustomBaggingClassifier(n_estimators=selected_B, max_depth=frozen_d, min_samples_leaf=frozen_m, random_state=config["seed"])
    final_bagger.fit(X_train, y_train)
    
    # Clean evaluation
    tree_clean_preds = final_tree.predict(X_test_clean)
    tree_clean_acc = compute_accuracy(y_test, tree_clean_preds)
    tree_clean_conf = compute_confidence(final_tree.predict_proba(X_test_clean))

    bag_clean_preds = final_bagger.predict(X_test_clean)
    bag_clean_acc = compute_accuracy(y_test, bag_clean_preds)
    bag_clean_conf = compute_confidence(final_bagger.predict_proba(X_test_clean))

    # Masked evaluation
    tree_masked_preds = final_tree.predict(X_test_masked)
    tree_masked_acc = compute_accuracy(y_test, tree_masked_preds)
    tree_masked_conf = compute_confidence(final_tree.predict_proba(X_test_masked))

    bag_masked_preds = final_bagger.predict(X_test_masked)
    bag_masked_acc = compute_accuracy(y_test, bag_masked_preds)
    bag_masked_conf = compute_confidence(final_bagger.predict_proba(X_test_masked))

    # Case Study Searches
    # 1. Bagging corrects single tree error
    corrected_indices = np.where((tree_clean_preds != y_test) & (bag_clean_preds == y_test))[0]
    
    # 2. Both models make mistakes
    both_wrong_indices = np.where((tree_clean_preds != y_test) & (bag_clean_preds != y_test))[0]

    case_studies = {
        "bagging_corrected_tree": [],
        "both_wrong": []
    }

    for idx in corrected_indices[:2]:
        votes = final_bagger.predict_vote_counts(X_test_clean[idx:idx+1])[0].tolist()
        case_studies["bagging_corrected_tree"].append({
            "test_sample_index": int(idx),
            "true_label": int(y_test[idx]),
            "tree_prediction": int(tree_clean_preds[idx]),
            "bagging_vote_counts": votes
        })

    for idx in both_wrong_indices[:2]:
        votes = final_bagger.predict_vote_counts(X_test_clean[idx:idx+1])[0].tolist()
        case_studies["both_wrong"].append({
            "test_sample_index": int(idx),
            "true_label": int(y_test[idx]),
            "tree_prediction": int(tree_clean_preds[idx]),
            "bagging_prediction": int(bag_clean_preds[idx]),
            "bagging_vote_counts": votes
        })

    # Prepare JSON Output
    output = {
        "frozen_tree_params": {
            "max_depth": frozen_d,
            "min_samples_leaf": frozen_m
        },
        "baseline_sweep": baseline_sweep_results,
        "bagging_sweep": bagging_sweep_results,
        "selected_ensemble_size": selected_B,
        "robustness_experiment": {
            "single_tree": {
                "clean_acc": float(tree_clean_acc),
                "clean_confidence": float(tree_clean_conf),
                "masked_acc": float(tree_masked_acc),
                "masked_confidence": float(tree_masked_conf)
            },
            "bagging": {
                "clean_acc": float(bag_clean_acc),
                "clean_confidence": float(bag_clean_conf),
                "masked_acc": float(bag_masked_acc),
                "masked_confidence": float(bag_masked_conf)
            }
        },
        "case_studies": case_studies
    }

    os.makedirs("logs", exist_ok=True)
    save_json(output, "logs/q1_results.json")
    print("\nQ1 completed successfully. Results saved to logs/q1_results.json")

if __name__ == "__main__":
    run_q1()