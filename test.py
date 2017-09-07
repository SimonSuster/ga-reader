import sys
import numpy as np
import cPickle as pkl
import shutil

from config import *
from model import GAReader
from utils import Helpers, DataPreprocessor, MiniBatchLoader

def main(load_path, params, mode='test'):

    nhidden = params['nhidden']
    dropout = params['dropout']
    word2vec = params['word2vec']
    dataset = params['dataset']
    nlayers = params['nlayers']
    train_emb = params['train_emb']
    char_dim = params['char_dim']
    use_feat = params['use_feat']
    gating_fn = params['gating_fn']

    if dataset == "clicr":
        dp = DataPreprocessor.DataPreprocessorClicr()
        data = dp.preprocess(
            "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/",
            no_training_set=True)
    else:
        dp = DataPreprocessor.DataPreprocessor()
        if dataset == "cnn":
            dataset = "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/CNN_DailyMail/cnn/questions/"
        elif dataset == "wdw":
            dataset = "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/wdw/"
        data = dp.preprocess(dataset, no_training_set=True)
    inv_vocab = data.inv_dictionary

    print("building minibatch loaders ...")
    if mode=='test':
        batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.test, BATCH_SIZE)
    else:
        batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.validation, BATCH_SIZE)

    print("building network ...")
    W_init, embed_dim = Helpers.load_word2vec_embeddings(data.dictionary[0], word2vec)
    m = GAReader.Model(nlayers, data.vocab_size, data.num_chars, W_init, 
            nhidden, embed_dim, dropout, train_emb, 
            char_dim, use_feat, gating_fn, save_attn=True)
    m.load_model('%s/best_model.p'%load_path)

    print("testing ...")
    pr = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_num_cand)).astype('float32')
    fids, attns = [], []
    total_loss, total_acc, n = 0., 0., 0
    for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, fnames in batch_loader_test:
        outs = m.validate(dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl)
        loss, acc, probs = outs[:3]
        attns += [[fnames[0],probs[0,:]] + [o[0,:,:] for o in outs[3:]]] # store one attention

        pred_ans = []
        for f in range(len(fnames)):
            pred_cand = probs[f].argmax()
            cand_pos = [c for c, i in enumerate(m_c[f]) if i == 1]
            pred_cand_pos = cand_pos[pred_cand]
            pred_cand_id = dw[f, [pred_cand_pos], 0]  # n, doc_id, 1
            pred_ent_anonym = inv_vocab[pred_cand_id]
            pred_ent_name = data.test_relabeling_dicts[fnames[f]][pred_ent_anonym]
            pred_ans.append(pred_ent_name)

        bsize = dw.shape[0]
        total_loss += bsize*loss
        total_acc += bsize*acc

        pr[n:n+bsize,:] = probs
        fids += fnames
        n += bsize

        #Helpers.show_predicted_vs_ground_truth(probs, a, )

    logger = open(load_path+'/log','a',0)
    message = '%s Loss %.4e acc=%.4f' % (mode.upper(), total_loss/n, total_acc/n)
    print message
    logger.write(message+'\n')
    logger.close()

    np.save('%s/%s.probs' % (load_path,mode),np.asarray(pr))
    pkl.dump(attns, open('%s/%s.attns' % (load_path,mode),'w'))
    f = open('%s/%s.ids' % (load_path,mode),'w')
    for item in fids: f.write(item+'\n')
    f.close()
