import numpy as np
from tqdm import tqdm
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../..")
from ecit import *



def ecitSimulate(methods, 
                 zDis_list=['gaussian','laplace'], 
                 noiseDis_list =['t4','t3','t2'], 
                 n_list=[800,1200,1600], 
                 t=1000):
    
    results_table = []

    for n in n_list:
        for zDis in zDis_list:
            for cit, k, p_ensemble in methods:
                row =  [n, zDis.capitalize(), cit.__name__ + str(k) + p_ensemble.__name__]

                for noiseDis in tqdm(noiseDis_list, desc=f"{n:>5}, {zDis:>8}, {k:>3},{p_ensemble.__name__:>11}"):
                    eI = 0
                    eII = 0
                    for i in range(t):
                        retries = 0
                        while retries < 5:
                            try:
                                dataI = np.hstack((generate_samples(n=n,indp='C',z_dis=zDis, noise_dis=noiseDis,noise_std=1)))
                                dataII = np.hstack((generate_samples(n=n,indp='N',z_dis=zDis, noise_dis=noiseDis,noise_std=1)))
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
                            eI += 1
                        if pII>0.05:
                            eII += 1
                    eI = eI/t
                    power = (t - eII)/t
                    row.append(eI)
                    row.append(power) 
                results_table.append(row)

    return results_table





def ecitSimulateDZ(methods,
                   zRange = range(1,6),
                   zDis='gaussian', 
                   noiseDis ='t3',
                   n=1200,
                   t=1000):
    
    results = {}

    for cit, k, p_ensemble in methods:

        eI_list = []
        power_list = []
        tim_list = []

        for dz in zRange:
            if cit != kcit or k!=1 or n<6000:
                start_time = time.time()
                eI = 0
                eII = 0
                for i in range(t):

                    retries = 0
                    while retries < 5:
                        try:
                            dataI = np.hstack((generate_samples(n=n,dz=dz,indp='C',z_dis=zDis, noise_dis=noiseDis,noise_std=1)))
                            dataII = np.hstack((generate_samples(n=n,dz=dz,indp='N',z_dis=zDis, noise_dis=noiseDis,noise_std=1)))
                            obj_ECIT = ECIT(dataI, cit, p_ensemble, k if k<100 else int(n/k))
                            pI = obj_ECIT([0], [1], list(range(2,dz+2)))
                            obj_ECIT = ECIT(dataII, cit, p_ensemble, k if k<100 else int(n/k))
                            pII = obj_ECIT([0], [1], list(range(2,dz+2)))
                            break
                        except Exception as e:
                            retries += 1
                            print(f"Retries times {retries}")
                            if retries >= 5: raise e
                    if pI<0.05:
                        eI += 1
                    if pII>0.05:
                        eII += 1
                eI = eI/t
                power = (t - eII)/t
                end_time = time.time()

                eI_list.append(eI)
                power_list.append(power)
                tim_list.append(end_time - start_time)

        
        results[cit.__name__ + str(k) + p_ensemble.__name__] = [eI_list, power_list, tim_list]

    return results




def plot_dz(results, save=False):

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


    dz = list(range(1,len(results[list(results.keys())[0]][0])+1))

    alpha = 0.95
    markersize = 3.8
    linewidth = 1.45

    fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.4), sharey=True, dpi=500)

    linestyles = ['--', '-', '-.']
    markers = ['^', 's', 'D']
    colors = ["#cf444d", sns.color_palette("muted")[0], sns.color_palette("muted")[9]]  # 颜色

    for i, key in enumerate(results.keys()):
        axes[0].plot(dz, results[key][0], alpha=alpha, label=label_name(key), linestyle=linestyles[i], marker=markers[i], markersize=markersize, linewidth=linewidth, color=colors[i])
        axes[1].plot(dz, results[key][1], alpha=alpha, label=label_name(key), linestyle=linestyles[i], marker=markers[i], markersize=markersize, linewidth=linewidth, color=colors[i])



    ax_eI, ax_eII = axes

    ax_eI.set_xticks(dz)
    ax_eII.set_xticks(dz)
    ax_eI.set_ylim(0, 1)
    ax_eII.set_ylim(0, 1)
    ax_eI.tick_params(axis='both', labelsize=9, pad=-2)
    ax_eII.tick_params(axis='both', labelsize=9, pad=-2)
    
    ax_eI.axhline(y=0.05, color='black', linestyle='-', alpha=0.6, linewidth = 0.5)

    ax_eI.set_title("Type I Error", fontsize=11)
    ax_eI.set_ylabel("Error Rate", fontsize=10)

    ax_eII.set_title("Power", fontsize=11)
    ax_eII.set_ylabel("Power", fontsize=10)


    ax_eII.legend(loc='lower right', fontsize=8.2, ncol=1)


    ax_eI.set_xlabel("Dimension of Z", fontsize=10)
    ax_eII.set_xlabel("Dimension of Z", fontsize=10)


    plt.tight_layout()
    if save: plt.savefig("plot_time.pdf", format='pdf')
    plt.show()

