import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from memory_profiler import memory_usage

from mlxtend.frequent_patterns import (
    apriori,
    fpgrowth,
    association_rules
)

from mlxtend.preprocessing import TransactionEncoder

from ifri_mini_ml_lib.association_rules.apriori import Apriori
from ifri_mini_ml_lib.association_rules.eclat import ECLAT
from ifri_mini_ml_lib.association_rules.fp_growth import FPGrowth
from ifri_mini_ml_lib.association_rules.association_rules import AssociationRules


# PREPROCESSING

def prepare_mlxtend_data(transactions):
    """
    Convert transaction list to one-hot encoded DataFrame
    compatible with mlxtend.
    """

    te = TransactionEncoder()

    te_ary = te.fit(transactions).transform(transactions)

    return pd.DataFrame(
        te_ary,
        columns=te.columns_
    )


# IFRI IMPLEMENTATIONS

def run_apriori(data, min_support, min_conf, min_lift):

    model = Apriori(
        min_support=min_support,
        min_confidence=min_conf
    )

    model.fit(data)

    frequent_itemsets = model.get_frequent_itemsets()

    assoc = AssociationRules(
        min_confidence=min_conf,
        min_lift=min_lift
    )

    return assoc.generate_rules(frequent_itemsets)


def run_eclat(data, min_support, min_conf, min_lift):

    model = ECLAT(
        min_support=min_support,
        min_confidence=min_conf
    )

    model.fit(data)

    frequent_itemsets = model.get_frequent_itemsets()

    assoc = AssociationRules(
        min_confidence=min_conf,
        min_lift=min_lift
    )

    return assoc.generate_rules(frequent_itemsets)


def run_fpgrowth(data, min_support, min_conf, min_lift):

    model = FPGrowth(
        min_support=min_support,
        min_confidence=min_conf
    )

    model.fit(data)

    frequent_itemsets = model.get_frequent_itemsets()

    assoc = AssociationRules(
        min_confidence=min_conf,
        min_lift=min_lift
    )

    return assoc.generate_rules(frequent_itemsets)


# MLXTEND IMPLEMENTATIONS

def run_mlxtend_apriori(df, min_support, min_conf, min_lift):

    frequent_itemsets = apriori(
        df,
        min_support=min_support,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_conf
    )

    rules = rules[rules["lift"] >= min_lift]

    return rules.reset_index(drop=True)


def run_mlxtend_fpgrowth(df, min_support, min_conf, min_lift):

    frequent_itemsets = fpgrowth(
        df,
        min_support=min_support,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_conf
    )

    rules = rules[rules["lift"] >= min_lift]

    return rules.reset_index(drop=True)



# MEMORY PROFILING


def run_with_memory(model_func, data, params):
    """
    Run model while measuring real process memory.
    """

    mem_usage, result = memory_usage(
        (
            model_func,
            (data,),
            params
        ),
        retval=True,
        interval=0.01,
        max_iterations=1
    )

    peak_memory = max(mem_usage)

    return result, peak_memory


# BENCHMARK CORE

def evaluate_model(
    model_name,
    model_func,
    data,
    n_runs=5,
    measure_memory=True,
    **params
):
    """
    Benchmark a model:
    - execution time
    - memory usage
    - rule statistics
    """

    times = []
    memories = []

    rule_counts = []
    confidences = []
    lifts = []

    
    # WARM-UP RUN

    try:
        model_func(data, **params)
    except Exception:
        pass

    
    # BENCHMARK RUNS

    for _ in range(n_runs):

        
        # TIME + MEMORY

        if measure_memory:

            start = time.perf_counter()

            result, peak_memory = run_with_memory(
                model_func,
                data,
                params
            )

            end = time.perf_counter()

        else:

            start = time.perf_counter()

            result = model_func(data, **params)

            end = time.perf_counter()

            peak_memory = np.nan

        
        # STORE METRICS

        exec_time = end - start

        times.append(exec_time)

        memories.append(peak_memory)

        rule_counts.append(len(result))

        
        # RULE METRICS

        if not result.empty:

            if "confidence" in result.columns:

                confidences.append(
                    result["confidence"].mean()
                )

            if "lift" in result.columns:

                lifts.append(
                    result["lift"].mean()
                )

    
    # FINAL RESULTS

    return {

        "model": model_name,

        "avg_time_sec": np.mean(times),

        "std_time_sec": np.std(times),

        "min_time_sec": np.min(times),

        "max_time_sec": np.max(times),

        "avg_memory_MB": (
            np.mean(memories)
            if measure_memory else np.nan
        ),

        "peak_memory_MB": (
            np.max(memories)
            if measure_memory else np.nan
        ),

        "avg_n_rules": np.mean(rule_counts),

        "avg_confidence": (
            np.mean(confidences)
            if confidences else 0
        ),

        "avg_lift": (
            np.mean(lifts)
            if lifts else 0
        )
    }



# GLOBAL BENCHMARK

def benchmark_all(
    transactions,
    min_support=0.02,
    min_conf=0.5,
    min_lift=1.0,
    n_runs=5,
    measure_memory=True
):
    """
    Benchmark all algorithms.
    """

    results = []

    
    # PREPARE DATA

    mlxtend_data = prepare_mlxtend_data(transactions)

    
    # MODELS

    models = [

        (
            "IFRI Apriori",
            run_apriori,
            transactions
        ),

        (
            "IFRI ECLAT",
            run_eclat,
            transactions
        ),

        (
            "IFRI FP-Growth",
            run_fpgrowth,
            transactions
        ),

        (
            "MLxtend Apriori",
            run_mlxtend_apriori,
            mlxtend_data
        ),

        (
            "MLxtend FP-Growth",
            run_mlxtend_fpgrowth,
            mlxtend_data
        )
    ]

    
    # BENCHMARK LOOP

    for name, func, data in models:

        result = evaluate_model(
            model_name=name,
            model_func=func,
            data=data,
            n_runs=n_runs,
            measure_memory=measure_memory,
            min_support=min_support,
            min_conf=min_conf,
            min_lift=min_lift
        )

        results.append(result)

    
    # FINAL DATAFRAME

    results_df = pd.DataFrame(results)

    
    # SPEEDUP

    baseline_time = results_df.iloc[0]["avg_time_sec"]

    results_df["speedup_vs_ifri_apriori"] = (
        baseline_time / results_df["avg_time_sec"]
    )

    
    # EFFICIENCY SCORE

    results_df["efficiency_score"] = (
        results_df["avg_lift"] /
        results_df["avg_time_sec"]
    )

    return results_df



# PLOT RESULTS

def plot_results(df):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 12)
    )

    
    # EXECUTION TIME

    axes[0, 0].bar(
        df["model"],
        df["avg_time_sec"]
    )

    axes[0, 0].set_title("Average Execution Time")

    axes[0, 0].set_ylabel("Time (sec)")

    axes[0, 0].tick_params(
        axis='x',
        rotation=20
    )

    
    # MEMORY

    axes[0, 1].bar(
        df["model"],
        df["avg_memory_MB"]
    )

    axes[0, 1].set_title("Average Memory Usage")

    axes[0, 1].set_ylabel("Memory (MB)")

    axes[0, 1].tick_params(
        axis='x',
        rotation=20
    )

    
    # RULE COUNT

    axes[1, 0].bar(
        df["model"],
        df["avg_n_rules"]
    )

    axes[1, 0].set_title("Average Number of Rules")

    axes[1, 0].set_ylabel("Rules")

    axes[1, 0].tick_params(
        axis='x',
        rotation=20
    )

    
    # SPEEDUP

    axes[1, 1].bar(
        df["model"],
        df["speedup_vs_ifri_apriori"]
    )

    axes[1, 1].set_title("Speedup vs IFRI Apriori")

    axes[1, 1].set_ylabel("Speedup")

    axes[1, 1].tick_params(
        axis='x',
        rotation=20
    )

    plt.tight_layout()

    plt.show()



# RULE SET COMPARISON

def compare_rule_sets(rules1, rules2):
    """
    Compare two rule sets using Jaccard similarity.
    """

    set1 = set(
        zip(
            rules1["antecedents"].astype(str),
            rules1["consequents"].astype(str)
        )
    )

    set2 = set(
        zip(
            rules2["antecedents"].astype(str),
            rules2["consequents"].astype(str)
        )
    )

    intersection = len(set1 & set2)

    union = len(set1 | set2)

    similarity = (
        intersection / union
        if union > 0 else 0
    )

    return {

        "intersection": intersection,

        "union": union,

        "jaccard_similarity": similarity
    }



# SCALABILITY BENCHMARK

def benchmark_scalability(
    model_name,
    model_func,
    datasets,
    min_support=0.02,
    min_conf=0.5,
    min_lift=1.0,
    measure_memory=False
):
    """
    Evaluate scalability with increasing dataset sizes.
    """

    results = []

    for data in datasets:

        start = time.perf_counter()

        if measure_memory:

            _, peak_memory = run_with_memory(
                model_func,
                data,
                {
                    "min_support": min_support,
                    "min_conf": min_conf,
                    "min_lift": min_lift
                }
            )

        else:

            model_func(
                data,
                min_support=min_support,
                min_conf=min_conf,
                min_lift=min_lift
            )

            peak_memory = np.nan

        end = time.perf_counter()

        results.append({

            "model": model_name,

            "dataset_size": len(data),

            "time_sec": end - start,

            "memory_MB": peak_memory
        })

    return pd.DataFrame(results)



# SCALABILITY PLOT

def plot_scalability(df):

    plt.figure(figsize=(10, 6))

    for model in df["model"].unique():

        subset = df[
            df["model"] == model
        ]

        plt.plot(
            subset["dataset_size"],
            subset["time_sec"],
            marker='o',
            label=model
        )

    plt.xlabel("Dataset Size")

    plt.ylabel("Execution Time (sec)")

    plt.title("Algorithm Scalability")

    plt.legend()

    plt.grid(True)

    plt.show()