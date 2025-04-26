import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
sys.path.append("../..")
from ecit import *



def ecitSimulateTime(n_list=[800,1600,2400,3200,4000],
                     t=1000,
                     zDis='gaussian',
                     noiseDis='t3'):
    
    results = {}

    for cit, k, p_ensemble in [(kcit, 1, p_alpha2), (rcit, 1, p_alpha2), (kcit, 400, p_alpha175), (kcit, 400, p_alpha2)]:

        eI = [0]*len(n_list)
        eII = [0]*len(n_list)
        tim = [0]*len(n_list)

        for i, n in enumerate(n_list):

            start_time = time.time()
            iteration_loop = tqdm(range(t), desc=f"{cit.__name__}, n={n}", leave=True, dynamic_ncols=True)
            for _ in iteration_loop:
                retries = 0
                while retries < 5:
                    try:
                        dataI = np.hstack((generate_samples(n=n,indp='C',z_dis=zDis,noise_dis=noiseDis,noise_std=1)))
                        dataII = np.hstack((generate_samples(n=n,indp='N',z_dis=zDis,noise_dis=noiseDis,noise_std=1)))

                        obj_ECIT = ECIT(dataI, cit, p_ensemble, k if k<100 else int(n/k))
                        pI = obj_ECIT([0], [1], [2])
                        obj_ECIT = ECIT(dataII, cit, p_ensemble, k if k<100 else int(n/k))
                        pII = obj_ECIT([0], [1], [2])
                        break
                    except Exception as e:
                        retries += 1
                        print(f"Retries times {retries}")
                        if retries >= 5: raise e
                if pI<0.05:
                    eI[i] += 1
                if pII>0.05:
                    eII[i] += 1

            end_time = time.time()
            tim[i] = end_time - start_time
            eI[i] = eI[i]/t
            eII[i] = eII[i]/t

        results[cit.__name__ + str(k) + p_ensemble.__name__] = [eI, eII, tim]

    return results








def show_results(results, n_list=[800,1600,2400,3200,4000], yl=0.66):
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


    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=1000, sharex=True)
    ax_eI, ax_eII, ax_tim = axes
    linestyles = ['--', ':', '-', '-.']
    markers = ['^', 'o', 's', 'D']
    colors = ["#cf444d", "#ff6969", sns.color_palette("muted")[0], sns.color_palette("muted")[9]]
    alphas = [0.95, 1, 0.95, 0.95]
    markersize = 4.2
    linewidth = 1.3

    for i, key in enumerate(results.keys()):

        eI, eII, tim = results[key]
        power = 1 - np.array(eII)
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        alpha = alphas[i % len(alphas)]

        x_vals = n_list[:len(eI)] if len(eI) != len(n_list) else n_list

        ax_eI.plot(x_vals, eI, alpha=alpha, label=label_name(key), linestyle=linestyle, marker=marker, markersize=markersize, linewidth=linewidth, color=color)
        ax_eII.plot(x_vals, power, alpha=alpha, label=label_name(key), linestyle=linestyle, marker=marker, markersize=markersize, linewidth=linewidth, color=color)
        ax_tim.plot(x_vals, np.array(tim)/500, alpha=alpha, label=label_name(key), linestyle=linestyle, marker=marker, markersize=markersize, linewidth=linewidth, color=color)

    ax_eI.set_title("Type I Error", fontsize=12)
    ax_eI.set_ylabel("Error Rate", fontsize=11)

    ax_eII.set_title("Power", fontsize=12)
    ax_eII.set_ylabel("Power", fontsize=11)
    ax_eII.set_xlabel("Sample Size (n)", fontsize=11)   

    ax_tim.set_title("Execution Time", fontsize=12)
    ax_tim.set_ylabel("Time (s)", fontsize=11)
    ax_tim.set_xlabel("Sample Size (n)", fontsize=11)

    ax_tim.set_xticks(n_list)
    ax_eII.set_xticks(n_list)
    ax_eI.set_xticks(n_list)


    ax_eI.axhline(y=0.05, color='black', linestyle='-', alpha=0.4, linewidth = 0.6)

    for ax in [ax_eI, ax_eII, ax_tim]:
        ax.set_xticklabels(n_list, rotation=45)
        ax.tick_params(axis='x', which='major', pad=-3)
        #ax.spines['right'].set_visible(False)
        #ax.spines['top'].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        ax.tick_params(axis='both', which='both', width=0.9, length=6, labelsize=10)
        ax.tick_params(axis='x', which='both', labelsize=7)
        

    ax_eI.set_ylim(0, 0.42)
    ax_eI.set_yticks(np.arange(0, 0.45, 0.1))
    ax_eII.set_ylim(yl, 1.02)
    #ax_eII.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax_tim.set_ylim(-4, 53)
    ax_tim.set_yticks(np.arange(0, 53, 15))

    ax_tim.legend(loc='upper left', fontsize=8.5, ncol=1)


    # 自动调整布局，防止图例和图形重叠
    plt.tight_layout()
    plt.show()
