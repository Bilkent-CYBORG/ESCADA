import numpy as np
import pandas as pd
from os import path
import sys

libpath = './../../bgl_model/'
sys.path.append(libpath)

from main_model import *

sigma_noise = 5  # noise std. dev.
bg_omin = 70
bg_omax = 180
postbg_target = 112.5

pbg_tar_adult = 140
pbg_tar_child = 125
pbg_tar_adolescent = 110

diff_targets = True

global_tmi = 0
global_tmbg = 150
beta = 5
lipschitz = 150

# EXPERIMENT SETUP
num_total_event = 30 # we do 30 runs with different meal events for each patient.
num_event = 1  # num. of different meal events in each run per patient (1 since single meal scenario).
rec_per_event = 15  # num. of recommendations per meal event in each run per patient (15).
num_rec = num_event*rec_per_event  # num. of total recommendations

#  context variables.
cvars = ['patient', 'meal', 'bg_fasting', 'ins_escada', 'ins_sts', 'ins_taco', 'ins_ts', 'ins_calc', 'ESCADA', 'STS', 'TACO', 'TS', 'Calc.'] 
cvars_tc = ['patient', 'meal', 'bg_fasting', 'ins_escada_tc', 'ins_sts_tc', 'ins_calc_tc', 'ESCADA-TC', 'STS-TC', 'Tuned Calc.']  # tuned calculator

#  patient list.
all_patients = ['adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005',
                'adult#006', 'adult#007', 'adult#008', 'adult#009', 'adult#010',
                'adolescent#001', 'adolescent#002', 'adolescent#003', 'adolescent#004', 'adolescent#005',
                'adolescent#006', 'adolescent#007', 'adolescent#008', 'adolescent#009', 'adolescent#010',
                'child#001', 'child#002', 'child#003', 'child#004', 'child#005',
                'child#006', 'child#007', 'child#008', 'child#009', 'child#010']
                
child_patients = ['child#001', 'child#002', 'child#003', 'child#004', 'child#005', 'child#006', 'child#007', 'child#009', 'child#010']
normal_patients = ['adult#002', 'adult#003', 'adult#004', 'adult#006', 'adult#007', 'adult#008', 'adolescent#001', 'adolescent#003', 'adolescent#004', 'adolescent#005', 'adolescent#006', 'adolescent#009', 'adolescent#010']
hard_patients = ['adult#001', 'adult#005', 'adult#010', 'adolescent#007', 'child#008']
extreme_patients = ['adolescent#002', 'adolescent#008', 'adult#009']

# use these to correct extreme patient for all cases (tuning the calculator parameters manually)

cr_correct = {'adult#001':0, 'adult#002':0, 'adult#003':0, 'adult#004':0, 'adult#005':0,
              'adult#006':0, 'adult#007':0, 'adult#008':0, 'adult#009':2, 'adult#010':0,
              'adolescent#001': 0, 'adolescent#002': 0, 'adolescent#003': 5, 'adolescent#004': 5, 'adolescent#005': 3,
              'adolescent#006': 0, 'adolescent#007': 4, 'adolescent#008': 2.2, 'adolescent#009': 0, 'adolescent#010': 0,
              'child#001':0, 'child#002':0, 'child#003':0, 'child#004':0, 'child#005':0,
              'child#006':0, 'child#007':0, 'child#008':8, 'child#009':0, 'child#010':0}

cf_correct = {'adult#001':4, 'adult#002':0, 'adult#003':0, 'adult#004':0, 'adult#005':0,
              'adult#006':0, 'adult#007':0, 'adult#008':0, 'adult#009':0, 'adult#010':0,
              'adolescent#001': 0, 'adolescent#002': 0, 'adolescent#003': 0, 'adolescent#004': 0, 'adolescent#005': 0,
              'adolescent#006': 0, 'adolescent#007': 0, 'adolescent#008': 4, 'adolescent#009': 0, 'adolescent#010': 0,
              'child#001':0, 'child#002':0, 'child#003':0, 'child#004':0, 'child#005':0,
              'child#006':0, 'child#007':0, 'child#008':9, 'child#009':0, 'child#010':0}

## use these to make sure the pump always makes safe recommendations (tuning the calculator parameters manually)

cr_correct_tc = {'adult#001':7, 'adult#002':1, 'adult#003':0, 'adult#004':0, 'adult#005':2,
                 'adult#006':4, 'adult#007':0, 'adult#008':3, 'adult#009':3, 'adult#010':3,
                 'adolescent#001': 0, 'adolescent#002': 3.5, 'adolescent#003': 8, 'adolescent#004': 8, 'adolescent#005': 5,
                 'adolescent#006': 0, 'adolescent#007': 6, 'adolescent#008': 2.9, 'adolescent#009': 8, 'adolescent#010': 4,
                 'child#001':-10, 'child#002':7, 'child#003':8, 'child#004':-10, 'child#005':-3,
                 'child#006':7, 'child#007':0, 'child#008':11, 'child#009':-20, 'child#010':5}

cf_correct_tc = {'adult#001':0, 'adult#002':0, 'adult#003':0, 'adult#004':0, 'adult#005':0,
                 'adult#006':0, 'adult#007':0, 'adult#008':0, 'adult#009':0, 'adult#010':0,
                 'adolescent#001': 0, 'adolescent#002': 0, 'adolescent#003': 0, 'adolescent#004': 0, 'adolescent#005': 0,
                 'adolescent#006': 0, 'adolescent#007': 0, 'adolescent#008': 7, 'adolescent#009': 0, 'adolescent#010': 0,
                 'child#001':-30, 'child#002':0, 'child#003':0, 'child#004':-20, 'child#005':0,
                 'child#006':0, 'child#007':0, 'child#008':22, 'child#009':0, 'child#010':0}

cr_correct_tc_dt = {'adult#001':7, 'adult#002':2, 'adult#003':2, 'adult#004':0, 'adult#005':2,
                 'adult#006':4, 'adult#007':4, 'adult#008':5, 'adult#009':3, 'adult#010':3,
                 'adolescent#001': 0, 'adolescent#002': 3.5, 'adolescent#003': 8, 'adolescent#004': 8, 'adolescent#005': 5,
                 'adolescent#006': 0, 'adolescent#007': 6, 'adolescent#008': 2.8, 'adolescent#009': 8, 'adolescent#010': 4,
                 'child#001':-8, 'child#002':9, 'child#003':9, 'child#004':-9, 'child#005':-3,
                 'child#006':8, 'child#007':0, 'child#008':11.5, 'child#009':-18, 'child#010':6}

cf_correct_tc_dt = {'adult#001':0, 'adult#002':0, 'adult#003':0, 'adult#004':0, 'adult#005':0,
                 'adult#006':0, 'adult#007':0, 'adult#008':0, 'adult#009':0, 'adult#010':0,
                 'adolescent#001': 0, 'adolescent#002': 0, 'adolescent#003': 0, 'adolescent#004': 0, 'adolescent#005': 0,
                 'adolescent#006': 0, 'adolescent#007': 0, 'adolescent#008': 7, 'adolescent#009': 0, 'adolescent#010': 0,
                 'child#001':-30, 'child#002':0, 'child#003':0, 'child#004':-20, 'child#005':0,
                 'child#006':0, 'child#007':0, 'child#008':23, 'child#009':0, 'child#010':0}

# to read simulator outputs from saved data

aux_dict = {'adult#001':10, 'adult#002':10, 'adult#003':10, 'adult#004':10, 'adult#005':10,
              'adult#006':10, 'adult#007':10, 'adult#008':10, 'adult#009':5, 'adult#010':10,
              'adolescent#001': 10, 'adolescent#002': 5, 'adolescent#003': 10, 'adolescent#004': 10, 'adolescent#005': 10,
              'adolescent#006': 10, 'adolescent#007': 10, 'adolescent#008': 5, 'adolescent#009': 10, 'adolescent#010': 10,
              'child#001':20, 'child#002':20, 'child#003':20, 'child#004':20, 'child#005':20,
              'child#006':20, 'child#007':20, 'child#008':10, 'child#009':20, 'child#010':20}

ins_dict = {'adult#001':35, 'adult#002':20, 'adult#003':20, 'adult#004':20, 'adult#005':35,
              'adult#006':20, 'adult#007':20, 'adult#008':20, 'adult#009':80, 'adult#010':35,
              'adolescent#001': 20, 'adolescent#002': 80, 'adolescent#003': 20, 'adolescent#004': 20, 'adolescent#005': 20,
              'adolescent#006': 20, 'adolescent#007': 35, 'adolescent#008': 80, 'adolescent#009': 20, 'adolescent#010': 20,
              'child#001':10, 'child#002':10, 'child#003':10, 'child#004':10, 'child#005':10,
              'child#006':10, 'child#007':10, 'child#008':35, 'child#009':10, 'child#010':10}

df = pd.read_csv('./../Quest.csv')

def create_model(data_init, cons, ins, bgs, ins_ceiling, target_bg):
    data_batch = data_init.loc[0:1, :]
    cons = cons[0:1]
    ins = ins[0:1]
    bgs = bgs[0:1]

    variables = [{'name': 'meal', 'domain': (20, 80), 'linearbounds':(0,9), 'unit': 'g'},
            {'name': 'insulin', 'domain': (0, ins_ceiling), 'linearbounds':(-9,0), 'unit': 'U'},
            {'name': 'bg_fasting', 'domain': (50, 400), 'linearbounds':(0,9), 'unit': 'mg/dl'}]

    X_init, Y_init = np.zeros((data_batch.shape[0],len(variables))), np.zeros((data_batch.shape[0],1))
    for i in range(data_batch.shape[0]):
        X_init[i] = [data_batch['meal'][i], data_batch['insulin'][i], data_batch['bg_fasting'][i]]
        Y_init[i] = [data_batch['bg_postprandial'][i]]

    bgl_model = BGL_MODEL(X_init, Y_init, postBG_target_range=(bg_omin, bg_omax), postBG_target_value=target_bg, variables=variables)

    bgl_model.update(cons, ins, bgs)

    return bgl_model

def generate_data(patient, meal_ind, cvars):
    all_data = pd.DataFrame(np.zeros((num_total_event,len(cvars))),columns=cvars)
    data = pd.DataFrame(np.zeros((num_rec,len(cvars))),columns=cvars)
    
    all_data.loc[:,'patient'] = patient
    np.random.seed(2)
    all_data.loc[:,'meal'] = np.random.randint(60, size=num_total_event) + 20 #(20,80)
    np.random.seed(3)
    all_data.loc[:,'bg_fasting'] = np.random.randint(50, size=num_total_event) + 100  # [100,150]

    for i in range(rec_per_event):
        data.loc[i,'patient'] = all_data.loc[meal_ind, 'patient']
        data.loc[i,'meal'] = all_data.loc[meal_ind, 'meal']
        data.loc[i,'bg_fasting'] = all_data.loc[meal_ind, 'bg_fasting']

    return (data, all_data)

def get_init_data(cur_patient, N_init, gt, cr, cf, ins_ceiling, data, bgl_vals, ins_spacing):
    cvars_init = ['patient', 'meal', 'insulin', 'bg_fasting', 'bg_postprandial']
    data_init = pd.DataFrame(np.zeros((N_init, len(cvars_init))),columns=cvars_init)
    c_list = []
    i_list = []
    bg_list = []
    for i in range(N_init):
        data_init.loc[i,'patient'] = cur_patient
        meal_ind = np.random.randint(num_total_event)
        data_init.loc[i,'meal'] = data['meal'][meal_ind] 
        data_init.loc[i,'bg_fasting'] = data['bg_fasting'][meal_ind] 
        insulin = np.minimum(std_bolus_calc(cho=data_init.loc[i,'meal'], gm=data_init.loc[i,'bg_fasting'], gt=gt, cr=cr, cf=cf), ins_ceiling - 0.1)
        data_init.loc[i,'insulin'] = insulin
        data_init.loc[i,'bg_postprandial'] = bgl_vals[meal_ind, int(insulin/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)

        c_list.append({'meal': data_init.loc[i, 'meal'], 'bg_fasting': data_init.loc[i, 'bg_fasting']})
        i_list.append(data_init.loc[i,'insulin'])
        bg_list.append(data_init.loc[i,'bg_postprandial'])

    return (data_init, c_list, i_list, bg_list)


# SME with tuned calculator

if diff_targets:
    cr_vals = {pt: df.loc[df['Name']==pt]['CR'].values[0] - cr_correct_tc_dt[pt] for pt in all_patients}
    cf_vals = {pt: df.loc[df['Name']==pt]['CF'].values[0] - cf_correct_tc_dt[pt] for pt in all_patients}
else:
    cr_vals = {pt: df.loc[df['Name']==pt]['CR'].values[0] - cr_correct_tc[pt] for pt in all_patients}
    cf_vals = {pt: df.loc[df['Name']==pt]['CF'].values[0] - cf_correct_tc[pt] for pt in all_patients}


pm_data_dict = {}
p_data_dict = {}

## FIX THE PATIENT AND MEAL EVENT FOR A RUN, RESULTS FROM ALL RUNS WILL BE AVERAGED.
for cur_patient in all_patients:

    bgl_vals = np.load('./../calc_res/{}_calc_res.npy'.format(cur_patient))

    if cur_patient in child_patients:
        ins_ceiling = 10
        aux_ins = 20
    elif cur_patient in normal_patients:
        ins_ceiling = 20
        aux_ins = 10
    elif cur_patient in hard_patients:
        ins_ceiling = 35
        aux_ins = 10
    else:
        ins_ceiling = 80
        aux_ins = 5

    ins_spacing = 1/aux_ins

    cr = cr_vals[cur_patient]
    cf = cf_vals[cur_patient]

    if diff_targets:
        if 'adult' in cur_patient:
            gt = pbg_tar_adult
        elif 'adolescent' in cur_patient:
            gt = pbg_tar_adolescent
        else:
            gt = pbg_tar_child
    else:
        gt = postbg_target  

    ## GENERATE INITIAL DATA just to initialize the model
    N_init = 2
    _, data_pre = generate_data(patient=cur_patient, meal_ind=1, cvars=cvars)
    data_init, cons, ins, bgs = get_init_data(cur_patient=cur_patient, N_init=N_init, gt=gt, cr=cr, cf=cf,
                                             ins_ceiling=ins_ceiling, data=data_pre, bgl_vals=bgl_vals, ins_spacing=ins_spacing)

    for meal_ind in range(num_total_event):
        print('\nPatient: {}, Meal Event: {}, Tuned Calc.'.format(cur_patient, meal_ind + 1))

        data_exp, _ = generate_data(patient=cur_patient, meal_ind=meal_ind, cvars= cvars_tc)

        bgl_model_escada = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling, target_bg=gt)
        bgl_model_sts = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling, target_bg=gt)

        for i in range(num_rec):

            context={'meal':data_exp['meal'][i], 
                     'bg_fasting':data_exp['bg_fasting'][i]}

            update = True
            if i == 0:
                ins_calc = std_bolus_calc(cho=context['meal'], gm=context['bg_fasting'], gt=gt, cr=cr, cf=cf)
                ins_calc = np.minimum(ins_calc, ins_ceiling - 0.1)
                pre_bgl_calc = bgl_vals[meal_ind, int(ins_calc/ins_spacing)]

            ins_escada = bgl_model_escada.recommend(context, strategy='escada', ins_calc=ins_calc, aux_ins=aux_ins, update=update, beta=beta, lipschitz=lipschitz)
            ins_sts = bgl_model_sts.recommend(context, strategy='sts', ins_calc=ins_calc, aux_ins=aux_ins, update=update, beta=beta, lipschitz=lipschitz)

            bgl_escada = bgl_vals[meal_ind, int(ins_escada/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)
            bgl_sts = bgl_vals[meal_ind, int(ins_sts/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)
            bgl_calc = pre_bgl_calc + np.random.normal(loc=0, scale=sigma_noise)

            data_exp.loc[i,'ins_escada_tc'] = ins_escada
            data_exp.loc[i,'ins_sts_tc'] = ins_sts
            data_exp.loc[i,'ins_calc_tc'] = ins_calc
            data_exp.loc[i,'ESCADA-TC'] = bgl_escada
            data_exp.loc[i,'STS-TC'] = bgl_sts
            data_exp.loc[i,'Tuned Calc.'] = bgl_calc
              
            # update
            bgl_model_escada.update(context, ins_escada, bgl_escada)
            bgl_model_sts.update(context, ins_sts, bgl_sts)

        data_exp.to_csv('./ppbg_tc_dt/{}_meal{}_sme.csv'.format(cur_patient, meal_ind + 1), index=False)
        pm_data_dict['{}_meal{}'.format(cur_patient, meal_ind + 1)] = data_exp

    p_data_dict[cur_patient] = pd.concat(list(dict(filter(lambda item: cur_patient in item[0], pm_data_dict.items())).values()), ignore_index=True)
    p_data_dict[cur_patient].to_csv('./ppbg_tc_dt/{}_sme.csv'.format(cur_patient), index=False)

data = pd.concat(list(p_data_dict.values()), ignore_index=True)
data.to_csv('./ppbg_tc_dt/tc_sme.csv', index=False)

# SME

cr_vals = {pt: df.loc[df['Name']==pt]['CR'].values[0] - cr_correct[pt] for pt in all_patients}
cf_vals = {pt: df.loc[df['Name']==pt]['CF'].values[0] - cf_correct[pt] for pt in all_patients}

pm_data_dict = {}
p_data_dict = {}

## FIX THE PATIENT AND MEAL EVENT FOR A RUN, RESULTS FROM ALL RUNS WILL BE AVERAGED.
for cur_patient in all_patients:

    bgl_vals = np.load('./../calc_res/{}_calc_res.npy'.format(cur_patient))

    if cur_patient in child_patients:
        ins_ceiling = 10
        aux_ins = 20
    elif cur_patient in normal_patients:
        ins_ceiling = 20
        aux_ins = 10
    elif cur_patient in hard_patients:
        ins_ceiling = 35
        aux_ins = 10
    else:
        ins_ceiling = 80
        aux_ins = 5

    ins_spacing = 1/aux_ins

    cr = cr_vals[cur_patient]
    cf = cf_vals[cur_patient]

    if diff_targets:
        if 'adult' in cur_patient:
            gt = pbg_tar_adult
        elif 'adolescent' in cur_patient:
            gt = pbg_tar_adolescent
        else:
            gt = pbg_tar_child
    else:
        gt = postbg_target  

    ## GENERATE INITIAL DATA just to initialize the model
    N_init = 2
    _, data_pre = generate_data(patient=cur_patient, meal_ind=1, cvars= cvars)
    data_init, cons, ins, bgs = get_init_data(cur_patient=cur_patient, N_init=N_init, gt=gt, cr=cr, cf=cf,
                                             ins_ceiling=ins_ceiling, data=data_pre, bgl_vals=bgl_vals, ins_spacing=ins_spacing)

    for meal_ind in range(num_total_event):
        print('\nPatient: {}, Meal Event: {}, Normal Calc.'.format(cur_patient, meal_ind + 1))

        data_exp, _ = generate_data(patient=cur_patient, meal_ind=meal_ind, cvars= cvars)

        bgl_model_escada = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling, target_bg=gt)
        bgl_model_sts = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling, target_bg=gt)
        bgl_model_taco = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling, target_bg=gt)
        bgl_model_ts = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling, target_bg=gt)

        for i in range(num_rec):

            context={'meal':data_exp['meal'][i], 
                     'bg_fasting':data_exp['bg_fasting'][i]}

            update = True
            if i == 0:
                ins_calc = std_bolus_calc(cho=context['meal'], gm=context['bg_fasting'], gt=gt, cr=cr, cf=cf)
                ins_calc = np.minimum(ins_calc, ins_ceiling - 0.1)
                pre_bgl_calc = bgl_vals[meal_ind, int(ins_calc/ins_spacing)]

            ins_escada = bgl_model_escada.recommend(context, strategy='escada', ins_calc=ins_calc, aux_ins=aux_ins, update=update, beta=beta, lipschitz=lipschitz)
            ins_sts = bgl_model_sts.recommend(context, strategy='sts', ins_calc=ins_calc, aux_ins=aux_ins, update=update, beta=beta, lipschitz=lipschitz)
            ins_taco = bgl_model_taco.recommend(context, strategy='taco', aux_ins=aux_ins, beta=beta)          
            ins_ts = bgl_model_ts.recommend(context, strategy='ts', aux_ins=aux_ins, beta=beta)

            bgl_escada = bgl_vals[meal_ind, int(ins_escada/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)
            bgl_sts = bgl_vals[meal_ind, int(ins_sts/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)
            bgl_taco = bgl_vals[meal_ind, int(ins_taco/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)
            bgl_ts = bgl_vals[meal_ind, int(ins_ts/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)
            bgl_calc = pre_bgl_calc + np.random.normal(loc=0, scale=sigma_noise)

            data_exp.loc[i,'ins_escada'] = ins_escada
            data_exp.loc[i,'ins_sts'] = ins_sts
            data_exp.loc[i,'ins_taco'] = ins_taco
            data_exp.loc[i,'ins_ts'] = ins_ts
            data_exp.loc[i,'ins_calc'] = ins_calc
            data_exp.loc[i,'ESCADA'] = bgl_escada
            data_exp.loc[i,'STS'] = bgl_sts
            data_exp.loc[i,'TACO'] = bgl_taco
            data_exp.loc[i,'TS'] = bgl_ts
            data_exp.loc[i,'Calc.'] = bgl_calc
              
            # update
            bgl_model_escada.update(context, ins_escada, bgl_escada)
            bgl_model_sts.update(context, ins_sts, bgl_sts)
            bgl_model_taco.update(context, ins_taco, bgl_taco)
            bgl_model_ts.update(context, ins_ts, bgl_ts)

        data_exp.to_csv('./ppbg_dt/{}_meal{}_sme.csv'.format(cur_patient, meal_ind + 1), index=False)
        pm_data_dict['{}_meal{}'.format(cur_patient, meal_ind + 1)] = data_exp

    p_data_dict[cur_patient] = pd.concat(list(dict(filter(lambda item: cur_patient in item[0], pm_data_dict.items())).values()), ignore_index=True)
    p_data_dict[cur_patient].to_csv('./ppbg_dt/{}_sme.csv'.format(cur_patient), index=False)

data = pd.concat(list(p_data_dict.values()), ignore_index=True)
data.to_csv('./ppbg_dt/nc_sme.csv', index=False)