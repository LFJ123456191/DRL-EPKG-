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

    # smooth data
    for datum in all_files:
        datum['reward']  =  datum['episodic_reward'].ewm(span = 3000).mean()
        datum['success']  =  datum['success_rate'].ewm(span = 3000).mean()
        datum  =  datum[datum['step']%50 == 0]
        processed_files.append(datum)

    # put data together
    data  =  pd.concat(processed_files)
    return data


# load and process data
parser  =  argparse.ArgumentParser()
parser.add_argument("scenario", help = "scenario to plot")          # left_turn
parser.add_argument('metric', help = 'metric to plot')              # success
args  =  parser.parse_args()

results_path = os.path.abspath('..') + '/train_results/' + args.scenario
path_1 = os.path.abspath('..')

# path = f'train_results/{args.scenario}'

files = os.listdir(results_path)
all_files = [os.path.join(results_path, _path) for _path in files]

# plot
plt.figure(figsize = (12, 8))

axes  =  plt.gca()
axes.set_xlim([0, 1e5])
axes.set_xlabel('Step', fontsize = 18)
plt.tick_params(labelsize = 16)

print('args.scenario =', args.scenario)
if args.scenario == 'roundabout_40':
    title_name = 'Roundabout'
elif args.scenario == 'left_turn_40':
    title_name = 'Left_turn'
elif args.scenario == '4lane_sv_40':
    title_name = 'Intersection'


axes.set_title(f'{title_name}', fontsize = 18)

if args.metric  ==  'success':
    axes.set_ylim([0, 1])
    axes.set_ylabel('Average success rate', fontsize = 18)
elif args.metric  ==  'reward' and args.scenario  ==  'left_turn':
    axes.set_ylim([-1, 2])
    axes.set_ylabel('Average episode reward')
elif args.metric  ==  'reward' and args.scenario  ==  'roundabout':
    axes.set_ylim([-1, 3])
    axes.set_ylabel('Average episode reward')
else:
    raise Exception('Undefined metric!')

labels_1 = []

for path_name in all_files:

    algo_name = os.path.basename(path_name)
    print('algo_name =', algo_name)
    if algo_name == 'Ours':
        continue
    labels_1.append(algo_name)

    results  =  load_and_process(path_1 + '/'+ f'train_results/{args.scenario}/{algo_name}')

    seaborn.set(style = "whitegrid", font_scale = 4, rc = {"lines.linewidth": 3})       # 风格设置，改变线的粗细
    seaborn.lineplot(data = results, x = 'step', y = args.metric, err_style = 'band')   # 折线图

algo_name = 'Ours'
labels_1.append(algo_name)
results  =  load_and_process(path_1 + '/'+ f'train_results/{args.scenario}/{algo_name}')
seaborn.set(style = "whitegrid", font_scale = 4, rc = {"lines.linewidth": 3})       # 风格设置，改变线的粗细
seaborn.lineplot(data = results, x = 'step', y = args.metric, err_style = 'band')   # 折线图

plt.legend(labels = labels_1, fontsize = 14)
print('Saving figure...', path_1)
plt.savefig(path_1 + "/draw_results/" + args.scenario + ".jpg")
plt.show()
print('Done!')
