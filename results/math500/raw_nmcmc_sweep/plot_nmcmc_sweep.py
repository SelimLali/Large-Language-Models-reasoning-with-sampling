import matplotlib.pyplot as plt

def main():
    nmcmc = [1, 2, 3, 4]
    acc =  [0.6666666, 0.7, 0.7, 0.7333333]
    acc_pct = [100.0 * a for a in acc]

    plt.figure(figsize=(6, 4))
    plt.plot(nmcmc, acc_pct, marker="o", label="Qwen/Qwen2.5-Math-1.5B-Instruct")

    plt.xlabel("MCMC Steps")
    plt.ylabel("MATH500 accuracy (%)")
    plt.title("Effect of varying N_MCMC hyperparameter of Power MH sampling")
    plt.grid(True)
    plt.legend()

    out_path = "results/math500/raw_nmcmc_sweep/nmcmc_sweep_plot.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved plot -> {out_path}")

if __name__ == "__main__":
    main()