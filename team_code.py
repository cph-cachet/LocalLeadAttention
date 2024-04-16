#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np
import os
import sys
import joblib
import pickle
from model_code import *
from model.blocks import FinalModel
from scipy import signal
from scipy.stats import zscore
from scipy.optimize import differential_evolution
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import shutil
from tqdm import tqdm
import warnings
import pandas as pd
import random
from skmultilearn.model_selection import iterative_train_test_split
from datetime import datetime
import copy
from pathlib import Path
from evaluate_model import load_weights, compute_challenge_metric
np.random.seed(0)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
bsize = 16
model_path = None

################################################################################
#
# Training function
#
################################################################################


class optim_genetics:
    def __init__(self, target, outputs, classes):
        self.target = target
        self.outputs = outputs
        weights_file = './weights.csv'
        self.normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'],
                              ['284470004', '63593006'],
                              ['427172004', '17338001'],
                              ['733534002', '164909002']]

        # Load the scored classes and the weights for the Challenge metric.
        print('Loading weights...')
        _, self.weights = load_weights(weights_file)
        self.classes = classes
        # match classed ordering
        # reorder = [self.classes.index(c) for c in classes]
        # self.outputs = self.outputs[:, reorder]
        # self.target = self.target[:, reorder]
        stop = 1

    def __call__(self, x):
        outputs = copy.deepcopy(self.outputs)
        outputs = outputs > x
        outputs = np.array(outputs, dtype=int)
        return -compute_challenge_metric(self.weights, self.target, outputs, self.classes, self.normal_class)


def find_thresholds(filename, model_directory):
    with open(filename, 'rb') as handle:
        models = pickle.load(handle)
        train_files = pickle.load(handle)
        valid_files = pickle.load(handle)
        classes = pickle.load(handle)
        lossweights = pickle.load(handle)

    results = pd.DataFrame(models)
    results.drop(columns=['model'], inplace=True)

    model_idx = np.argmax(results[:]['valid_auprc'])
    t = results.iloc[model_idx]['valid_targets']
    y = results.iloc[model_idx]['valid_outputs']

    N = 26
    f1prcT = np.zeros((N,))
    f1rocT = np.zeros((N,))

    for j in range(N):
        prc, rec, thr = precision_recall_curve(
            y_true=t[:, j], probas_pred=y[:, j])
        fscore = 2 * prc * rec / (prc + rec)
        idx = np.nanargmax(fscore)
        f1prc = np.nanmax(fscore)
        f1prcT[j] = thr[idx]

        fpr, tpr, thr = roc_curve(y_true=t[:, j], y_score=y[:, j])
        fscore = 2 * (1 - fpr) * tpr / (1 - fpr + tpr)
        idx = np.nanargmax(fscore)
        f1roc = np.nanmax(fscore)
        f1rocT[j] = thr[idx]

    population = np.random.rand(300, N)
    for i in range(1, 99):
        population[i, :] = i / 100

    print(f1prcT)
    print(f1rocT)
    population[100] = f1rocT
    population[101] = f1prcT
    bounds = [(0, 1) for i in range(N)]

    result = differential_evolution(optim_genetics(
        t, y, classes), bounds=bounds, disp=True, init=population, workers=-1)
    print(result)
    select4deployment(models[model_idx]['model'], thresholds=result.x,
                      classes=classes, info='', model_directory=model_directory)


def select4deployment(state_dict, thresholds, classes, info, model_directory):
    select4deployment.calls += 1
    name = Path(model_directory, f'MODEL_{select4deployment.calls}.pickle')
    with open(name, 'wb') as handle:
        model = FinalModel(num_classes=26)
        model.load_state_dict(state_dict)
        model.cpu()
        model.eval()

        pickle.dump({'state_dict': model.state_dict(),
                     'classes': classes,
                     'thresholds': thresholds,
                     'info': info}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


class mytqdm(tqdm):
    def __init__(self, dataset):
        super(mytqdm, self).__init__(dataset, ncols=0)
        self.alpha = 0.99
        self._val = None

    def set_postfix(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.data.cpu().numpy()
        if self._val is None:
            self._val = loss
        else:
            self._val = self.alpha*self._val + (1-self.alpha)*loss
        super(mytqdm, self).set_postfix({'loss': self._val})


class challengeloss(nn.Module):
    def __init__(self):
        super(challengeloss, self).__init__()
        weights_file = './weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'],
                              ['284470004', '63593006'],
                              ['427172004', '17338001'],
                              ['733534002', '164909002']]

        # Load the scored classes and the weights for the Challenge metric.
        print('Loading weights...')
        load_classes, self.weights = load_weights(weights_file)
        self.weights = torch.from_numpy(self.weights).float().to(
            DEVICE).requires_grad_(False)
        self.I = torch.ones((26, 26)).float().to(DEVICE).requires_grad_(False)

    def forward(self, L, P):
        L = L.float()
        N = L + P - L * P
        N = torch.mm(N, self.I) + 1e-6
        C = torch.mm(L.T, P / N)
        C = torch.sum(self.weights * C)
        return C


def get_nsamp(header):
    return int(header.split('\n')[0].split(' ')[3])


class dataset:
    classes = ['164889003', '164890007', '6374002', '426627000', '733534002',
               '713427006', '270492004', '713426002', '39732003', '445118002',
               '164947007', '251146004', '111975006', '698252002', '426783006',
               '284470004', '10370003', '365413008', '427172004', '164917005',
               '47665007', '427393009', '426177001', '427084000', '164934002',
               '59931005']
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'],
                          ['284470004', '63593006'],
                          ['427172004', '17338001'],
                          ['733534002', '164909002']]

    def __init__(self, header_files):
        self.files = []
        self.sample = True
        self.num_leads = None
        for h in tqdm(header_files):
            tmp = dict()
            tmp['header'] = h
            tmp['record'] = h.replace('.hea', '.mat')
            hdr = load_header(h)
            tmp['nsamp'] = get_nsamp(hdr)
            tmp['leads'] = get_leads(hdr)
            tmp['age'] = get_age(hdr)
            tmp['sex'] = get_sex(hdr)
            tmp['dx'] = get_labels(hdr)
            tmp['fs'] = get_frequency(hdr)
            tmp['target'] = np.zeros((26,))
            tmp['dx'] = replace_equivalent_classes(
                tmp['dx'], dataset.equivalent_classes)
            for dx in tmp['dx']:
                # in SNOMED code is in scored classes
                if dx in dataset.classes:
                    idx = dataset.classes.index(dx)
                    tmp['target'][idx] = 1
            self.files.append(tmp)

        # set filter parameters
        self.b, self.a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

        self.files = pd.DataFrame(self.files)

    def train_valid_split(self, test_size):
        files = self.files['header'].to_numpy().reshape(-1, 1)
        targets = np.stack(self.files['target'].to_list(), axis=0)
        x_train, y_train, x_valid, y_valid = iterative_train_test_split(
            files, targets, test_size=test_size)
        train = dataset(header_files=x_train[:, 0].tolist())
        train.num_leads = None
        train.sample = True
        valid = dataset(header_files=x_valid[:, 0].tolist())
        valid.num_leads = 12
        valid.sample = False
        return train, valid

    def summary(self, output):
        if output == 'pandas':
            return pd.Series(np.stack(self.files['target'].to_list(), axis=0).sum(axis=0), index=dataset.classes)
        if output == 'numpy':
            return np.stack(self.files['target'].to_list(), axis=0).sum(axis=0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        fs = self.files.iloc[item]['fs']
        target = self.files.iloc[item]['target']
        leads = self.files.iloc[item]['leads']
        data = load_recording(self.files.iloc[item]['record'])

        # expand to 12 lead setup if original signal has less channels
        data, lead_indicator = expand_leads(data, input_leads=leads)
        data = np.nan_to_num(data)

        # resample to 500hz
        if fs == float(1000):
            data = signal.resample_poly(
                data, up=1, down=2, axis=-1)  # to 500Hz
            fs = 500
        elif fs == float(500):
            pass
        else:
            data = signal.resample(data, int(data.shape[1] * 500 / fs), axis=1)
            fs = 500

        data = signal.filtfilt(self.b, self.a, data)

        if self.sample:
            fs = int(fs)
            # random sample signal if len > 8192 samples
            if data.shape[-1] >= 8192:
                idx = data.shape[-1] - 8192-1
                idx = np.random.randint(idx)
                data = data[:, idx:idx + 8192]

        mu = np.nanmean(data, axis=-1, keepdims=True)
        std = np.nanstd(data, axis=-1, keepdims=True)
        # std = np.nanstd(data.flatten())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = (data - mu) / std
        data = np.nan_to_num(data)

        # random choose number of leads to keep
        data, lead_indicator = lead_exctractor.get(
            data, self.num_leads, lead_indicator)

        return data, target, lead_indicator


def expand_leads(recording, input_leads):
    output = np.zeros((12, recording.shape[1]))
    twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    twelve_leads = [k.lower() for k in twelve_leads]

    input_leads = [k.lower() for k in input_leads]
    output_leads = np.zeros((12,))
    for i, k in enumerate(input_leads):
        idx = twelve_leads.index(k)
        output[idx, :] = recording[i, :]
        output_leads[idx] = 1
    return output, output_leads


class lead_exctractor:
    """
    used to select specific leads or random choice of configurations

    Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Six leads: I, II, III, aVR, aVL, aVF
    Four leads: I, II, III, V2
    Three leads: I, II, V2
    Two leads: I, II

    """
    L2 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    L3 = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    L4 = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    L6 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    L8 = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    L12 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    @staticmethod
    def get(x, num_leads, lead_indicator):
        if num_leads == None:
            # random choice output
            num_leads = random.choice([12, 8, 6, 4, 3, 2])

        if num_leads == 12:
            # Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
            return x, lead_indicator * lead_exctractor.L12

        if num_leads == 8:
            # Six leads: I, II, III, aVL, aVR, aVF
            x = x * lead_exctractor.L8.reshape(12, 1)
            return x, lead_indicator * lead_exctractor.L8

        if num_leads == 6:
            # Six leads: I, II, III, aVL, aVR, aVF
            x = x * lead_exctractor.L6.reshape(12, 1)
            return x, lead_indicator * lead_exctractor.L6

        if num_leads == 4:
            # Six leads: I, II, III, V2
            x = x * lead_exctractor.L4.reshape(12, 1)
            return x, lead_indicator * lead_exctractor.L4

        if num_leads == 3:
            # Three leads: I, II, V2
            x = x * lead_exctractor.L3.reshape(12, 1)
            return x, lead_indicator * lead_exctractor.L3

        if num_leads == 2:
            # Two leads: II, V5
            x = x * lead_exctractor.L2.reshape(12, 1)
            return x, lead_indicator * lead_exctractor.L2
        raise Exception("invalid-leads-number")


def collate(batch):
    ch = batch[0][0].shape[0]
    # maxL = max([b[0].shape[-1] for b in batch])
    maxL = 8192
    X = np.zeros((len(batch), ch, maxL))
    for i in range(len(batch)):
        X[i, :, -batch[i][0].shape[-1]:] = batch[i][0]
    t = np.array([b[1] for b in batch])
    l = np.concatenate([b[2].reshape(1, 12) for b in batch], axis=0)

    X = torch.from_numpy(X)
    t = torch.from_numpy(t)
    l = torch.from_numpy(l)
    return X, t, l


# Adapted from original scoring function code
# For each set of equivalent classes, replace each class with the representative class for the set.
def replace_equivalent_classes(classes, equivalent_classes):
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
            if x in multiple_classes:
                # Use the first class as the representative class.
                classes[j] = multiple_classes[0]
    return classes


def valid_part(model, dataset):
    targets = []
    outputs = []
    weights_file = 'weights.csv'
    sinus_rhythm = set(['426783006'])

    classes, weights = load_weights(weights_file)
    model.eval()
    with torch.no_grad():
        for i, (x, t, l) in enumerate(tqdm(dataset)):
            x = x.unsqueeze(2).float().to(DEVICE)
            t = t.to(DEVICE)
            l = l.float().to(DEVICE)

            y = model(x, l)
            # p = torch.sigmoid(y)

            targets.append(t.data.cpu().numpy())
            outputs.append(y.data.cpu().numpy())
    targets = np.concatenate(targets, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    auprc = average_precision_score(y_true=targets, y_score=outputs)
    challenge_metric = compute_challenge_metric(
        weights, targets, outputs, classes, sinus_rhythm)
    return auprc, targets, outputs, challenge_metric


def train_part(model, dataset, loss, opt):
    targets = []
    outputs = []
    model.train()
    chloss = challengeloss()
    with mytqdm(dataset) as pbar:
        for i, (x, t, l) in enumerate(pbar):
            opt.zero_grad()

            x = x.unsqueeze(2).float().to(DEVICE)
            t = t.to(DEVICE)
            l = l.float().to(DEVICE)

            y = model(x, l)
            # p = torch.sigmoid(y)

      #      M = chloss(t, p)
      #      N = loss(input=p, target=t)
      #      Q = torch.mean(-4*p*(p-1))
      #      J = N - M + Q
            J = -torch.mean(t * F.logsigmoid(y) + (1 - t)
                            * F.logsigmoid(-y) * 0.1)
            J.backward()
            pbar.set_postfix(np.array([J.data.cpu().numpy(),
                                       ]))

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            opt.step()

            targets.append(t.data.cpu().numpy())
            outputs.append(y.data.cpu().numpy())
        targets = np.concatenate(targets, axis=0)
        outputs = np.concatenate(outputs, axis=0)
        auprc = average_precision_score(y_true=targets, y_score=outputs)
    return auprc


def training_code(data_directory, model_directory):
    select4deployment.calls = 0
    _training_code(data_directory, model_directory, str(0))
    _training_code(data_directory, model_directory, str(1))
    _training_code(data_directory, model_directory, str(2))


def _training_code(data_directory, model_directory, ensamble_ID):
    # Find header and recording files.
    print('Finding header and recording files...')

    # train,valid = load_k_fold(0,data_directory)
    header_files, recording_files = find_challenge_files(data_directory)

    full_dataset = dataset(header_files)
    print(full_dataset.summary('pandas'))
    train, valid = full_dataset.train_valid_split(test_size=0.05)

    valid.files = valid.files[valid.files['nsamp'] <= 8192]
    valid.files.reset_index(drop=True, inplace=True)

    # negative to positive ratio
    loss_weight = (len(train) - train.summary(output='numpy')) / \
        train.summary(output='numpy')

    # to be saved in resulting model pickle
    train_files = train.files['header'].to_list()
    train_files = [k.split('/')[-1] for k in train_files]
    valid_files = valid.files['header'].to_list()
    valid_files = [k.split('/')[-1] for k in valid_files]

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    train = DataLoader(dataset=train,
                       batch_size=bsize,
                       shuffle=True,
                       num_workers=8,
                       collate_fn=collate,
                       pin_memory=True,
                       drop_last=False)

    valid = DataLoader(dataset=valid,
                       batch_size=bsize,
                       shuffle=False,
                       num_workers=8,
                       collate_fn=collate,
                       pin_memory=True,
                       drop_last=False)

    model = FinalModel(num_classes=26).to(DEVICE)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print('Model Loaded!')

    lossBCE = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(loss_weight).to(DEVICE))
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    OUTPUT = []
    EPOCHS = 100
    for epoch in range(EPOCHS):
        print(
            f"============================[{epoch}]============================")
        train_auprc = train_part(model, train, lossBCE, opt)
        print(train_auprc)

        valid_auprc, valid_targets, valid_outputs, challenge_metric = valid_part(
            model, valid)
        print(valid_auprc)

        OUTPUT.append({'epoch': epoch,
                       'model': copy.deepcopy(model).cpu().state_dict(),
                       'train_auprc': train_auprc,
                       'valid_auprc': valid_auprc,
                       'valid_targets': valid_targets,
                       'valid_outputs': valid_outputs,
                       'val_challenge_metric': challenge_metric})
        scheduler.step()
        name = Path(model_directory, f'PROGRESS_{ensamble_ID}.pickle')
        with open(name, 'wb') as handle:
            pickle.dump(OUTPUT, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(valid_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(dataset.classes, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(loss_weight, handle, protocol=pickle.HIGHEST_PROTOCOL)

    name = Path(model_directory, f'PROGRESS_{ensamble_ID}.pickle')
    with open(name, 'wb') as handle:
        pickle.dump(OUTPUT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset.classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(loss_weight, handle, protocol=pickle.HIGHEST_PROTOCOL)

    find_thresholds(name, model_directory)


# Generic function for loading a model.
def _load_model(model_directory, id):
    filename = Path(model_directory, f'MODEL_{id}.pickle')
    model = {}
    with open(filename, 'rb') as handle:
        input = pickle.load(handle)

    model['classifier'] = FinalModel(num_classes=26).to(DEVICE)
    model['classifier'].load_state_dict(input['state_dict'])
    model['classifier'].eval()
    model['thresholds'] = input['thresholds']
    model['classes'] = input['classes']
    return model


def load_model(model_directory, leads):

    model = {}
    model['1'] = _load_model(model_directory, 1)
   # model['2'] = _load_model(model_directory, 2)
   # model['3'] = _load_model(model_directory, 3)
    return model


################################################################################
#
# Running trained model functions
#
################################################################################


def expand_leads(recording, input_leads):
    output = np.zeros((12, recording.shape[1]))
    twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    twelve_leads = [k.lower() for k in twelve_leads]

    input_leads = [k.lower() for k in input_leads]
    output_leads = np.zeros((12,))
    for i, k in enumerate(input_leads):
        idx = twelve_leads.index(k)
        output[idx, :] = recording[i, :]
        output_leads[idx] = 1
    return output, output_leads


def zeropad(x):
    y = np.zeros((12, 8192))
    if x.shape[1] < 8192:
        y[:, -x.shape[1]:] = x
    else:
        y = x[:, :8192]
    return y


def preprocessing(recording, leads, fs):
    b, a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

    if fs == 1000:
        recording = signal.resample_poly(
            recording, up=1, down=2, axis=-1)  # to 500Hz
        fs = 500
    elif fs == 500:
        pass
    else:
        recording = signal.resample(recording, int(
            recording.shape[1] * 500 / fs), axis=1)
        print(f'RESAMPLING FROM {fs} TO 500')
        fs = 500

    recording = signal.filtfilt(b, a, recording)
    recording = zscore(recording, axis=-1)
    recording = np.nan_to_num(recording)
    recording = zeropad(recording)
    recording = torch.from_numpy(recording).view(
        1, 12, 1, -1).float().to(DEVICE)
    leads = torch.from_numpy(leads).float().view(1, 12).to(DEVICE)
    return recording, leads


# Generic function for running a trained model.
def run_model(model, header, recording):
    # load lead names form file
    input_leads = get_leads(header)
    recording, leads = expand_leads(recording, input_leads)
    recording, leads = preprocessing(
        recording, leads, fs=get_frequency(header))

    classes = model['1']['classes']

    out_labels = np.zeros((3, 26))
    for i, (key, mod) in enumerate(model.items()):
        thresholds = mod['thresholds']

        classifier = mod['classifier']

        q = classifier(recording, leads)
        # p = torch.sigmoid(_probabilities)
        q = q.data[0, :].cpu().numpy()

        # Predict labels and probabilities.
        labels = q >= thresholds
        out_labels[i, :] = labels
    labels = np.sum(out_labels, axis=0)
    labels = np.array(labels, dtype=np.int)
    return classes, labels, q

################################################################################
#
# Other functions
#
################################################################################
