import os

#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import train
import test
import argparse
import numpy as np
import random

from config import get_params

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', dest='mode', type=int, default=0,
                    help='run mode - (0-train+test, 1-train only, 2-test only, 3-val only)')
parser.add_argument('--nlayers', dest='nlayers', type=int, default=3,
                    help='Number of reader layers')
parser.add_argument('--nhidden', dest='nhidden', type=int, default=128,
                    help='Number of hidden units')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2,
                    help='Dropout rate')
parser.add_argument('--use_feat', dest='use_feat', type=int, default=0,
                    help='use indicator feature (0-no, 1-yes)')
parser.add_argument('--dataset', dest='dataset', type=str, default='wdw',
                    help='Dataset - (cnn || dailymail || cbtcn || cbtne || wdw || clicr || clicr_plain)')
parser.add_argument('--seed', dest='seed', type=int, default=1,
                    help='Seed for different experiments with same settings')
parser.add_argument('--gating_fn', dest='gating_fn', type=str, default='T.mul',
                    help='Gating function (T.mul || Tsum || Tconcat)')
parser.add_argument('--relabeling', dest='relabeling', type=int, default=1,
                    help='use relabeling (anonymization) (0-no, 1-yes)')
args = parser.parse_args()
cmd = vars(args)
params = get_params(cmd['dataset'])
params.update(cmd)

np.random.seed(params['seed'])
random.seed(params['seed'])

# save directory
w2v_filename = params['word2vec'].split('/')[-1].split('.')[0] if params['word2vec'] else 'None'
save_path = ('experiments/' +
             #'/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/ga-reader/experiments/' +
             params['dataset'].split('/')[0] +
             '_nhid%d' % params['nhidden'] + '_nlayers%d' % params['nlayers'] +
             '_dropout%.1f' % params['dropout'] + '_%s' % w2v_filename + '_chardim%d' % params['char_dim'] +
             '_train%d' % params['train_emb'] +
             '_seed%d' % params['seed'] + '_use-feat%d' % params['use_feat'] +
             '_gf%s' % params['gating_fn'] +
             '_relab%d' % params['relabeling'] +
             '/')
if not os.path.exists(save_path): os.makedirs(save_path)

# train
if params['mode'] < 2:
    train.main(save_path, params)

# test
if params['mode'] == 0 or params['mode'] == 2:
    test.main(save_path, params)
elif params['mode'] == 3:
    test.main(save_path, params, mode='validation')
