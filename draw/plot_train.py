import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import glob
import argparse
import os

def load_and_process(exp):
    # load data
    # files  =  glob.glob('{}/*/training_log.csv'.format(exp))
    files  =  glob.glob('{}/*/training_log.csv'.format(exp))
    all_files  =  [pd.read_csv(file) for file in files] # ['step'] ['success rate'] ['episodic reward']
    processed_files  =  []
    print('all_files = ', len(all_files))
    # smooth data
    for datum in all_files:
        datum['reward']  =  datum['episodic_reward'].ewm(span = 3000).mean()
        datum['success']  =  datum['success_rate'].ewm(span = 3000).mean()
        datum  =  datum[datum['step']%50 == 0]
        processed_files.append(datum)

    # put data together
    data  =  pd.concat(processed_files)
    print('data = ', data.shape)
    return data

# load and process data
parser  =  argparse.ArgumentParser()
parser.add_argument("algo", help = "algorithm to plot")             # value_penalty
parser.add_argument("scenario", help = "scenario to plot")          # left_turn
parser.add_argument('metric', help = 'metric to plot')              # success
# parser.add_argument('file_name', help = '')              # success
args  =  parser.parse_args()

results  =  load_and_process(f'train_results/{args.scenario}/{args.algo}')

# plot
plt.figure(figsize = (10, 6))
seaborn.set(style = "whitegrid", font_scale = 2, rc = {"lines.linewidth": 3})       # 风格设置，改变线的粗细
seaborn.lineplot(data = results, x = 'step', y = args.metric, err_style = 'band')   # 折线图
axes  =  plt.gca()
axes.set_xlim([0, 1e5])
axes.set_xlabel('Step')

if args.metric  ==  'success':
    axes.set_ylim([0, 1])
    axes.set_ylabel('Average success rate')
elif args.metric  ==  'reward' and args.scenario  ==  'left_turn':
    axes.set_ylim([-1, 2])
    axes.set_ylabel('Average episode reward')
elif args.metric  ==  'reward' and args.scenario  ==  'roundabout':
    axes.set_ylim([-1, 3])
    axes.set_ylabel('Average episode reward')
else:
    raise Exception('Undefined metric!')

axes.set_title(f'{args.scenario} ({args.algo})')
plt.tight_layout()                                      # 避免多个图重叠
print('Saving figure...', os.getcwd())
plt.savefig(os.getcwd() + "/plot_results/" + args.algo +'_' + args.scenario +".jpg")
plt.show()
print('Done!')

