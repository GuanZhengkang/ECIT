import numpy as np
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append("../..")
from ecit import *


def epcSimulate(methods=[(kcit, 1, p_alpha2), (rcit, 1, p_alpha2), (kcit, 400, p_alpha175), (kcit, 400, p_alpha2)], 
                n_nodes = 5, dense = 0.3, t = 100, n_list = [800,1600,2400,3200,4000], df=3):

    results = {}

    for cit, k, p_ensemble in methods:

        result = []

        for n in n_list:
            start_time = time.time()
            f1 = []
            shd = []
            for _ in range(t):
                retries = 0
                while retries < 5:
                    try:
                        data, tcg = generate_graph_samples(n,n_nodes,dense,df=df)
                        cg = epc(data, cit, p_ensemble, k if k < 100 else int(n/k), show_progress=False)
                        cg = cg.G.graph
                        break
                    except Exception as e:
                        retries += 1
                        #print(f"Retries times {retries}")
                        if retries >= 5: raise e
                f1.append(compute_skeleton_f1(cg, tcg)[-1])
                shd.append(compute_skeleton_SHD(cg, tcg))
            end_time = time.time()
            result.append([np.mean(f1), np.std(f1), np.mean(shd), np.std(shd), end_time - start_time])
        
        results[cit.__name__ + str(k) + p_ensemble.__name__] = np.array(result).T.tolist()

    return results




def show_results(results, n_list=[800,1600,2400,3200,4000], t=100, save=False):
    sns.set()
    
    def label_name(s):
        def convert_alpha_string(s):
            match = re.search(r'alpha(\d+)', s)
            if match:
                raw_number = match.group(1)
                if len(raw_number) == 1:
                    number = raw_number
                elif len(raw_number) == 2:
                    number = raw_number[0] + '.' + raw_number[1]
                else:
                    number = raw_number[:-2] + '.' + raw_number[-2:]
                return fr'($\alpha = {number}$)'
            else: return s


        match = re.search(r'\d+', s)
        if match:
            before = s[:match.start()]
            if before=='fisherz': before = 'FisherZ'
            else: before = before.upper()
            number = match.group()
            after = s[match.end():]
            if number == '1': return before
            else: return 'E'+ before + ' ' + convert_alpha_string(after)
        else: return s


    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.85), dpi=500, sharex=True)
    ax_f1, ax_shd, ax_tim = axes
    linestyles = ['--', ':', '-', '-.']
    markers = ['^', 'o', 's', 'D']
    colors = ["#cf444d", "#ff6969", sns.color_palette("muted")[0], sns.color_palette("muted")[9]]
    alphas = [0.95, 1, 0.95, 0.95]
    markersize = 4.2
    linewidth = 1.3

    for i, key in enumerate(results.keys()):

        f1, f1_std, shd, shd_std, tim = results[key]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        alpha = alphas[i % len(alphas)]
        x_vals = n_list

        ax_f1.plot(x_vals, f1, alpha=alpha, label=label_name(key), linestyle=linestyle, marker=marker, markersize=markersize, linewidth=linewidth, color=color)
        ax_f1.fill_between(x_vals,
                    np.array(f1) - np.array(f1_std),
                    np.array(f1) + np.array(f1_std),
                    color=color, alpha=0.09)
        ax_shd.plot(x_vals, shd, alpha=alpha, label=label_name(key), linestyle=linestyle, marker=marker, markersize=markersize, linewidth=linewidth, color=color)
        ax_shd.fill_between(x_vals,
                    np.array(shd) - np.array(shd_std),
                    np.array(shd) + np.array(shd_std),
                    color=color, alpha=0.09)
        ax_tim.plot(x_vals, np.array(tim)/t, alpha=alpha, label=label_name(key), linestyle=linestyle, marker=marker, markersize=markersize, linewidth=linewidth, color=color)

    ax_f1.set_title("F1 Score", fontsize=13)
    ax_f1.set_ylabel("F1 Score", fontsize=12)

    ax_shd.set_title("SHD", fontsize=13)
    ax_shd.set_ylabel("SHD", fontsize=12)
    ax_shd.set_xlabel("Sample Size (n)", fontsize=12, labelpad=8)   

    ax_tim.set_title("Execution Time", fontsize=13)
    ax_tim.set_ylabel("Time (s)", fontsize=12)

    ax_tim.set_xticks(n_list)
    ax_shd.set_xticks(n_list)
    ax_f1.set_xticks(n_list)


    for ax in [ax_f1, ax_shd, ax_tim]:
        ax.set_xticklabels(n_list, rotation=45)
        ax.tick_params(axis='x', which='major', pad=-3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        ax.tick_params(axis='both', which='both', width=0.9, length=6, labelsize=10.2)
        ax.tick_params(axis='x', which='both', labelsize=8.2)
        

    ax_tim.legend(loc='upper left', fontsize=8.5, ncol=1)


    plt.tight_layout()
    if save: plt.savefig("plot_pc.pdf", format='pdf')
    plt.show()