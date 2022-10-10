import numpy as np
import pandas as pd
from os import path
import sys

libpath = './../../bgl_model/'
sys.path.append(libpath)

from main_model import *

sigma_noise = 5  # noise std. dev.
ins_ceiling = 45  # max. insulin dose 
ins_spacing = 0.1
bg_omin = 70  # hypo boundary
bg_omax = 180  # hyper boundary
postbg_target = 112.5  # target dose

global_tmi = 0.0  # we assume the patient takes the bolus dose with the meal
target_tmbg = 150  # TACO makes recommendations to optimize the PPBG after 150 mins from the meal
beta = 4
lipschitz = 70

num_all_train = 20  # num. of available training events per patient
num_all_test = 20  # num. of available test events per patient

all_patients = ['adult#002', 'adult#005', 'adult#007', 'adult#008', 'adult#010']

clinician_recs = {'adult#002': [6.0, 4.5, 7.2, 2.7, 4.1, 5,6.5, 5.6, 5.4, 4.5, 12.2, 6.1, 11.9, 6.9, 2.7, 5.6, 6.8, 8.8, 6.8, 5.9],
                  'adult#005': [2.9, 11.8, 17.3, 19.9, 15.8, 15.2, 8.8,15.4, 7.8, 3, 12.7, 12.2, 4.5, 11.9, 7.8, 10.9, 8.8,26.7, 9.8, 8.5],
                  'adult#007': [1.4, 2.8, 3.6, 2.8, 3.2, 1.8, 0.9, 3.8, 3.2, 3.6, 6.3, 2.6, 2, 2.2, 1.7, 2.9, 1, 3.5, 3.4, 2.5],
                  'adult#008': [2.7, 3.4, 2.4, 1.4, 4.3, 2.8, 2.2, 2.8, 2.2, 0.9, 1.1, 1.7, 3, 3, 0.7, 2.5, 4.4, 2.1, 5.5, 1.3,],
                  'adult#010': [5.4, 17.8, 12.3, 13.6, 16.1, 12.2, 8.2, 10, 22.9, 24.8, 6.7, 33.1, 41.5, 6.7, 4.8, 14.5, 11.2, 15.9, 18.9, 9]}


def create_model(data_init, cons, ins, bgs, ins_ceiling):
    data_batch = data_init.loc[0:1, :]

    variables = [{'name': 'meal', 'domain': (0, 400), 'linearbounds':(0,9), 'unit': 'g'},
             {'name': 'tmbg', 'domain': (120, 210), 'linearbounds':(-9,0), 'unit': 'min'},
             {'name': 'insulin', 'domain': (0, ins_ceiling), 'linearbounds':(-9,0), 'unit': 'U'},
             {'name': 'bg_fasting', 'domain': (50, 800), 'linearbounds':(0,9), 'unit': 'mg/dl'}]
             
    X_init, Y_init = np.zeros((data_batch.shape[0],len(variables))), np.zeros((data_batch.shape[0],1))
    for i in range(data_batch.shape[0]):
        X_init[i] = [data_batch['meal'][i], data_batch['tmbg'][i], data_batch['insulin'][i], data_batch['bg_fasting'][i]]
        Y_init[i] = [data_batch['bg_postprandial'][i]]

    bgl_model = BGL_MODEL(X_init, Y_init, postBG_target_range=(bg_omin, bg_omax), postBG_target_value=postbg_target, variables=variables)

    bgl_model.update(cons, ins, bgs)

    return bgl_model


def get_init_data(N_init, data_init):
    c_list = []
    i_list = []
    bg_list = []
    for i in range(N_init):
        c_list.append({'meal': data_init.loc[i, 'meal'], 'bg_fasting': data_init.loc[i, 'bg_fasting'], 'tmbg': data_init.loc[i, 'tmbg']})
        i_list.append(data_init.loc[i,'insulin'])
        bg_list.append(data_init.loc[i,'bg_postprandial'])

    return (c_list, i_list, bg_list)


num_train = 20  # num. of training events per patient to be used
num_test = 20 # num. of test events per patient

test_dict = {}  # dict to store test results for each patient

# 1. Initiate a GP model for a patient
# 2. Feed the training data for that patient to GP
# 3. Obtain and FIX the GP Posterior
# 4. Make recommendations to patients for test events, using the GP Posterior
# 5. Repeat the steps above for each patient in list "all_patients"

for cur_patient in all_patients:

    bgl_vals = np.load('./../calc_res_clinician_data/{}_calc_res.npy'.format(cur_patient))

    df = pd.read_csv('./../Quest.csv')
    cr = df.loc[df['Name']==cur_patient]['CR'].values[0] 
    cf = df.loc[df['Name']==cur_patient]['CF'].values[0]

    train_data = pd.read_csv('./train_data/train_{}.csv'.format(cur_patient))
    test_data = pd.read_csv('./test_data/test_{}.csv'.format(cur_patient))

    cons, ins, bgs = get_init_data(num_train, train_data)

    bgl_model = create_model(train_data, cons, ins, bgs, ins_ceiling=ins_ceiling)

    for i in range(num_test):
        print('\nPatient: {}, Meal Event: {}'.format(cur_patient, i + 1))

        context={'meal': test_data['meal'][i],
                 'tmbg': target_tmbg, 
                 'bg_fasting': test_data['bg_fasting'][i]}
            
        ins_calc = std_bolus_calc(cho=context['meal'], gm=context['bg_fasting'], gt=postbg_target, cr=cr, cf=cf)
        ins_escada = bgl_model.recommend(context, strategy='escada', ins_calc=ins_calc, aux_ins=10, update=True, beta=beta, lipschitz=lipschitz)
        ins_cli = clinician_recs[cur_patient][i]
     
        bgl_calculator = bgl_vals[i, int(ins_calc/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)
        bgl_escada = bgl_vals[i, int(ins_escada/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)
        bgl_cli = bgl_vals[i, int(ins_cli/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)

        test_data.loc[i,'insulin'] = ins_escada
        test_data.loc[i,'insulin_calc'] = ins_calc
        test_data.loc[i,'insulin_cli'] = ins_cli
        test_data.loc[i,'ESCADA'] = bgl_escada
        test_data.loc[i,'Calculator'] = bgl_calculator
        test_data.loc[i,'Clinicians'] = bgl_cli

    test_data.to_csv('./test_res/test_res_{}.csv'.format(cur_patient), index=False)
    test_dict['{}'.format(cur_patient)] = test_data

all_test = pd.concat(list(test_dict.values()), ignore_index=True)
all_test.to_csv('./test_res/test_res.csv', index=False)