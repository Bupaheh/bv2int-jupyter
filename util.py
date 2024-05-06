import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
from collections import defaultdict


def get_stat(series):
    res = {}
    res['mean'] = str(round(series.mean(), 2))
    res['std'] = str(round(series.std(), 2))
    qs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
    for k, v in series.quantile(qs).to_dict().items():
        res[str(round(k, 2))] = str(round(v, 2))
    return res

def get_diff_stat(series):
    res = {}
    res['win prob'] = str(round((series > 0).mean(), 2))
    res['lose mean'] = str(round(-(series[series <= 0].mean()), 2))
    res['win mean'] = str(round(series[series > 0].mean(), 2))
    res['lose std'] = str(round(series[series <= 0].std(), 2))
    res['win std'] = str(round(series[series > 0].std(), 2))
    return res

def aggregate_stats(l, f):
    res = defaultdict(list)
    for series in l:
        d = f(series)
        for k, v in d.items():
            res[k] = res[k] + [v]
    return res

def stats(d):
    time = d.values()
    names = d.keys()

    return pd.DataFrame(
        aggregate_stats(
            time,
            get_stat
        ),
        index=names
    )

def get_unsat_data(df):
	return df[config.unsat_idx.intersection(df.index)]

def get_sat_data(df):
	return df[config.sat_idx.intersection(df.index)]

def unsat_stats(d):
    return stats({ k : get_unsat_data(v) for k, v in d.items()})

def sat_stats(d):
    return stats({ k : get_sat_data(v) for k, v in d.items()})
            
def check(data):
    sat_err = None
    unsat_err = None
    
    correct = data[((data['solver'] == 'Yices') | (data['solver'] == 'Z3'))].drop_duplicates('exprId')[['exprId', 'status']].set_index('exprId')

    sat = correct[correct['status'] == 'SAT']
    unsat = correct[correct['status'] == 'UNSAT']

    normalized_data = data.set_index('exprId')[['solver', 'status']]

    sat_data = normalized_data.loc[normalized_data.index.intersection(sat.index)]
    sat_error = (sat_data['status'] == 'UNSAT').any()
    
    if sat_error:
        sat_err = sat_data[sat_data['status'] == 'UNSAT']
        print('sat error')
        print(sat_data[sat_data['status'] == 'UNSAT'])
        
    unsat_data = normalized_data.loc[normalized_data.index.intersection(unsat.index)]
    unsat_error = (unsat_data['status'] == 'SAT').any()
    
    if unsat_error:
        unsat_err = unsat_data[unsat_data['status'] == 'SAT']
        print('unsat error')
        print(unsat_data[unsat_data['status'] == 'SAT'])
        
    return sat_err, unsat_err
            
def dict_filter(d, p):
    return { k : v for k, v in d.items() if p(k) }

def plot_helper(data, names, qs, base=10):
    expr_count = min([len(d.index) for d in data])
    # colors = ['green', 'red']
    # for d, name, c in zip(data, names, colors):
    #     plt.plot(np.round(qs * expr_count), d.quantile(qs), label=name, color=c)

    for d, name in zip(data, names):
        plt.plot(np.round(qs * expr_count), d.quantile(qs), label=name)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('solving time (us)', fontsize="23")
    plt.xlabel('number of formulas', fontsize="23")
    #plt.legend(loc='upper left', fontsize="20")
    plt.legend(loc='upper left', fontsize="12")
    plt.yscale('log', base=base)
    plt.grid()

def plot(d, start=0.0, end=1.0, base=10):
    data = d.values()
    names = d.keys()
    #plt.figure(figsize=(18, 25))
    plt.figure(figsize=(18, 6))

    #plt.subplot(4, 1, 1)
    #qs = np.linspace(0, 1, 500)
    #plot_helper(data, names, qs)

    #plt.subplot(4, 1, 2)
    #qs = np.linspace(0, end, 500)
    #plot_helper(data, names, qs)


    #plt.subplot(4, 1, 3)
    #qs = np.linspace(start, 1, 500)
    #plot_helper(data, names, qs)
    
    #plt.subplot(4, 1, 4)
    qs = np.linspace(start, end, 2000)
    plot_helper(data, names, qs, base)
    # plt.savefig('foo.png', bbox_inches='tight')
    plt.show()

def unsat_plot(d, start=0.0, end=1.0, base=10):
    return plot({ k : get_unsat_data(v) for k, v in d.items()}, start, end, base)

def sat_plot(d, start=0.0, end=1.0, base=10):
    return plot({ k : get_sat_data(v) for k, v in d.items()}, start, end, base)

def save_plot(d, path):
    d = intersection(d)
    sat_d = { k : get_sat_data(v) for k, v in d.items()}
    unsat_d = { k : get_unsat_data(v) for k, v in d.items()}
    qs = np.linspace(0, 1, 500)

    sat_data = sat_d.values()
    sat_names = sat_d.keys()
    unsat_data = unsat_d.values()
    unsat_names = unsat_d.keys()
    
    plt.figure(figsize=(21, 12))

    plt.subplot(2, 1, 1)
    plt.title('SAT', fontsize=25)
    plot_helper(sat_data, sat_names, qs)

    plt.subplot(2, 1, 2)
    plt.title('UNSAT', fontsize=25)
    plot_helper(unsat_data, unsat_names, qs)

    plt.tight_layout(pad=1.5)
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    
def intersection(d):
    idx = None

    for _, v in d.items():
        if idx is not None:
            idx = idx.intersection(v.index)
        else:
            idx = v.index
            
    return { k : v[idx] for k, v in d.items() }

def series_intersection(s1, s2):
    idx = s1.index.intersection(s2.index)
            
    return s1[idx], s2[idx]

class Config:
	sat_idx = None
	unsat_idx = None
	
	def init(self, data):
		correct = data[(data['solver'] != 'Yices-Eager-Sum-SignedLazyOverflow-Round1Status') & (data['solver'] != 'Yices-Eager-Sum-SignedLazyOverflow-Round1Status-Split')]
		self.sat_idx = correct[(correct['status'] == 'SAT')].drop_duplicates('exprId').set_index('exprId').index
		self.unsat_idx = correct[(correct['status'] == 'UNSAT')].drop_duplicates('exprId').set_index('exprId').index
	
config = Config()
