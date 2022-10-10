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

global_tmi = 0
global_tmbg = 150
beta = 5
lipschitz = 150

# EXPERIMENT SETUP
num_event = 30  # num. of different meal events in each run per patient.
rec_per_event = 15  # num. of recommendations per meal event in each run per patient.
num_rec = num_event*rec_per_event  # num. of total recommendations

#  context variables.
cvars = ['patient', 'meal', 'bg_fasting', 'ins_taco', 'ins_gpucb1', 'ins_gpucb2', 'ins_gpucb3', 'ins_calc',
         'TACO', 'GP-UCB-1', 'GP-UCB-2', 'GP-UCB-3', 'Calc.', 'TACO-N', 'GP-UCB-1-N', 'GP-UCB-2-N', 'GP-UCB-3-N', 'Calc.-N']

#  patient list.
all_patients = ['adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005',
                'adult#006', 'adult#007', 'adult#008', 'adult#009', 'adult#010',
                'child#001', 'child#002', 'child#003', 'child#004', 'child#005', 
                'child#006', 'child#007', 'child#008', 'child#009', 'child#010',
                'adolescent#001', 'adolescent#002', 'adolescent#003', 'adolescent#004', 'adolescent#005', 
                'adolescent#006', 'adolescent#007', 'adolescent#008', 'adolescent#009', 'adolescent#010']

# use these to correct extreme patient for all cases

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

cr_vals = {pt: df.loc[df['Name']==pt]['CR'].values[0] - cr_correct[pt] for pt in all_patients}
cf_vals = {pt: df.loc[df['Name']==pt]['CF'].values[0] - cf_correct[pt] for pt in all_patients}

child_patients = ['child#001', 'child#002', 'child#003', 'child#004', 'child#005', 'child#006', 'child#007', 'child#009', 'child#010']

normal_patients = ['adult#002', 'adult#003', 'adult#004', 'adult#006', 'adult#007', 'adult#008', 'adolescent#001', 'adolescent#003', 'adolescent#004', 'adolescent#005', 'adolescent#006', 'adolescent#009', 'adolescent#010']

hard_patients = ['adult#001', 'adult#005', 'adult#010', 'adolescent#007', 'child#008']

extreme_patients = ['adolescent#002', 'adolescent#008', 'adult#009']

def create_model(data_batch, cons, ins, bgs, ins_ceiling):

    variables = [{'name': 'meal', 'domain': (20, 80), 'linearbounds':(0,9), 'unit': 'g'},
            {'name': 'insulin', 'domain': (0, ins_ceiling), 'linearbounds':(-9,0), 'unit': 'U'},
            {'name': 'bg_fasting', 'domain': (50, 400), 'linearbounds':(0,9), 'unit': 'mg/dl'}]

    X_init, Y_init = np.zeros((data_batch.shape[0],len(variables))), np.zeros((data_batch.shape[0],1))
    for i in range(data_batch.shape[0]):
        X_init[i] = [data_batch['meal'][i], data_batch['insulin'][i], data_batch['bg_fasting'][i]]
        Y_init[i] = [data_batch['bg_postprandial'][i]]

    bgl_model = BGL_MODEL(X_init, Y_init, postBG_target_range=(bg_omin, bg_omax), postBG_target_value=postbg_target, variables=variables)

    bgl_model.update(cons[0:1], ins[0:1], bgs[0:1])

    return bgl_model

def generate_data(patient):
    data = pd.DataFrame(np.zeros((num_rec,len(cvars))),columns=cvars)
    
    data.loc[:,'patient'] = patient
    np.random.seed(2)
    data.loc[:,'meal'] = np.random.randint(60, size=num_rec) + 20 #(20,80)
    np.random.seed(3)
    data.loc[:,'bg_fasting'] = np.random.randint(50, size=num_rec) + 100  # [100,150]

    for i in range(rec_per_event):
        l_ind = int(i*num_event)
        u_ind = int((i+1)*num_event - 1)
        data.loc[l_ind:u_ind,'meal'] = data.loc[0:num_event-1,'meal'].to_numpy()
        data.loc[l_ind:u_ind,'bg_fasting'] = data.loc[0:num_event-1,'bg_fasting'].to_numpy()

    return (data, data.loc[0:num_event-1,:])


def get_init_data(cur_patient, N_init, gt, cr, cf, data, bgl_vals, ins_spacing):
    cvars_init = ['patient', 'meal', 'insulin', 'bg_fasting', 'bg_postprandial']
    data_init = pd.DataFrame(np.zeros((N_init, len(cvars_init))),columns=cvars_init)
    c_list = []
    i_list = []
    bg_list = []
    for i in range(N_init):
        data_init.loc[i,'patient'] = cur_patient
        meal_ind = np.random.randint(num_event)
        data_init.loc[i,'meal'] = data['meal'][meal_ind] 
        data_init.loc[i,'bg_fasting'] = data['bg_fasting'][meal_ind] 
        insulin = np.minimum(std_bolus_calc(cho=data_init.loc[i,'meal'], gm=data_init.loc[i,'bg_fasting'], gt=gt, cr=cr, cf=cf), ins_ceiling - 0.1)
        data_init.loc[i,'insulin'] = insulin
        data_init.loc[i,'bg_postprandial'] = bgl_vals[meal_ind, int(insulin/ins_spacing)] + np.random.normal(loc=0, scale=sigma_noise)

        c_list.append({'meal': data_init.loc[i, 'meal'], 'bg_fasting': data_init.loc[i, 'bg_fasting']})
        i_list.append(data_init.loc[i,'insulin'])
        bg_list.append(data_init.loc[i,'bg_postprandial'])

    return (data_init, c_list, i_list, bg_list)


data_dict = {}
save_folder = 'ppbg'

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
    gt = postbg_target

    # generate experiment data
    data_exp, data_pre = generate_data(patient=cur_patient)

    ## GENERATE INITIAL DATA
    N_init = 2
    data_init, cons, ins, bgs = get_init_data(cur_patient=cur_patient, N_init=N_init, gt=gt, cr=cr, cf=cf,
                                              data=data_pre, bgl_vals=bgl_vals, ins_spacing=ins_spacing)

    bgl_model_escada = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling)
    bgl_model_sts = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling)
    bgl_model_taco = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling)
    bgl_model_ts = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling)

    bgl_model_gpucb1 = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling)
    bgl_model_gpucb2 = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling)
    bgl_model_gpucb3 = create_model(data_init, cons, ins, bgs, ins_ceiling=ins_ceiling)

    ins_calc_dict = {}
    bgl_calc_dict = {}

    ## SEQUENTIALLY LEARN
    for i in range(num_rec):
        meal_ind = i%num_event + 1
        meal_rec_num = i//num_event + 1
        if meal_ind == 30:
            print('\nPatient: {}, Rec. Num.: {}'.format(cur_patient, meal_rec_num))

        # context
        context={'meal':data_exp['meal'][i], 
                 'bg_fasting':data_exp['bg_fasting'][i]}

        if meal_rec_num == 1:
            ins_calc = std_bolus_calc(cho=context['meal'], gm=context['bg_fasting'], gt=gt, cr=cr, cf=cf)
            ins_calc = np.minimum(ins_calc, ins_ceiling - 0.1)
            bgl_calc = bgl_vals[meal_ind - 1, int(ins_calc/ins_spacing)]
            ins_calc_dict[meal_ind] = ins_calc
            bgl_calc_dict[meal_ind] = bgl_calc
        else:
            ins_calc = ins_calc_dict[meal_ind]
            bgl_calc = bgl_calc_dict[meal_ind]

        ins_escada = bgl_model_escada.recommend(context, strategy='escada', ins_calc=ins_calc, aux_ins=aux_ins, update=True, beta=beta, lipschitz=lipschitz)
        ins_sts = bgl_model_sts.recommend(context, strategy='sts', ins_calc=ins_calc, aux_ins=aux_ins, update=True, beta=beta, lipschitz=lipschitz)
        ins_taco = bgl_model_taco.recommend(context, strategy='taco', aux_ins=aux_ins, beta=beta)           
        ins_ts = bgl_model_ts.recommend(context, strategy='ts', aux_ins=aux_ins, beta=beta)
        
        ins_gpucb1 = bgl_model_gpucb1.recommend(context, strategy='gpucb', aux_ins=aux_ins, beta=beta)    
        ins_gpucb2 = bgl_model_gpucb2.recommend(context, strategy='gpucb', aux_ins=aux_ins, beta=beta)  
        ins_gpucb3 = bgl_model_gpucb3.recommend(context, strategy='gpucb', aux_ins=aux_ins, beta=beta)    

        data_exp.loc[i,'ins_escada'] = ins_escada
        data_exp.loc[i,'ins_sts'] = ins_sts
        data_exp.loc[i,'ins_taco'] = ins_taco
        data_exp.loc[i,'ins_ts'] = ins_ts

        data_exp.loc[i,'ins_gpucb1'] = ins_gpucb1
        data_exp.loc[i,'ins_gpucb2'] = ins_gpucb2
        data_exp.loc[i,'ins_gpucb3'] = ins_gpucb3
        
        data_exp.loc[i,'ins_calc'] = ins_calc 
        
        bgl_escada = bgl_vals[meal_ind - 1, int(ins_escada/ins_spacing)]
        bgl_sts = bgl_vals[meal_ind - 1, int(ins_sts/ins_spacing)]
        bgl_taco = bgl_vals[meal_ind - 1, int(ins_taco/ins_spacing)]
        bgl_ts = bgl_vals[meal_ind - 1, int(ins_ts/ins_spacing)]

        bgl_gpucb1 = bgl_vals[meal_ind - 1, int(ins_gpucb1/ins_spacing)]  
        bgl_gpucb2 = bgl_vals[meal_ind - 1, int(ins_gpucb2/ins_spacing)]  
        bgl_gpucb3 = bgl_vals[meal_ind - 1, int(ins_gpucb3/ins_spacing)]  
        
        bgl_calc = bgl_calc 

        data_exp.loc[i,'ESCADA-N'] = bgl_escada
        data_exp.loc[i,'STS-N'] = bgl_sts
        data_exp.loc[i,'TACO-N'] = bgl_taco
        data_exp.loc[i,'TS-N'] = bgl_ts

        data_exp.loc[i,'GP-UCB-1-N'] = bgl_gpucb1
        data_exp.loc[i,'GP-UCB-2-N'] = bgl_gpucb2
        data_exp.loc[i,'GP-UCB-3-N'] = bgl_gpucb3
        
        data_exp.loc[i,'Calc.-N'] = bgl_calc

        bgl_escada = bgl_escada + np.random.normal(loc=0, scale=sigma_noise)
        bgl_sts = bgl_sts + np.random.normal(loc=0, scale=sigma_noise)
        bgl_taco = bgl_taco + np.random.normal(loc=0, scale=sigma_noise)
        bgl_ts = bgl_ts + np.random.normal(loc=0, scale=sigma_noise)

        bgl_gpucb1 = bgl_gpucb1 + np.random.normal(loc=0, scale=sigma_noise)
        bgl_gpucb2 = bgl_gpucb2 + np.random.normal(loc=0, scale=sigma_noise)
        bgl_gpucb3 = bgl_gpucb3 + np.random.normal(loc=0, scale=sigma_noise)
        
        bgl_calc = bgl_calc + np.random.normal(loc=0, scale=sigma_noise)

        data_exp.loc[i,'ESCADA'] = bgl_escada
        data_exp.loc[i,'STS'] = bgl_sts
        data_exp.loc[i,'TACO'] = bgl_taco
        data_exp.loc[i,'TS'] = bgl_ts

        data_exp.loc[i,'GP-UCB-1'] = bgl_gpucb1
        data_exp.loc[i,'GP-UCB-2'] = bgl_gpucb2
        data_exp.loc[i,'GP-UCB-3'] = bgl_gpucb3
        
        data_exp.loc[i,'Calc.'] = bgl_calc

        # update
        bgl_model_escada.update(context, ins_escada, bgl_escada)
        bgl_model_sts.update(context, ins_sts, bgl_sts)
        bgl_model_taco.update(context, ins_taco, bgl_taco)
        bgl_model_ts.update(context, ins_ts, bgl_ts)

        reward_gpucb1 =  -np.log(np.abs(bgl_gpucb1 - postbg_target) + 1) 
        bgl_model_gpucb1.update(context, ins_gpucb1, reward_gpucb1)
        reward_gpucb2 = 1 - np.exp(np.abs(bgl_gpucb2 - postbg_target)/20)
        bgl_model_gpucb2.update(context, ins_gpucb2, reward_gpucb2)
        reward_gpucb3 = -np.abs(bgl_gpucb3 - postbg_target)
        bgl_model_gpucb3.update(context, ins_gpucb3, reward_gpucb3)
        

    data_exp.to_csv('./{}/{}_mme.csv'.format(save_folder, cur_patient))
    data_dict[cur_patient] = data_exp

data = pd.concat(list(data_dict.values()), ignore_index=True)
data.to_csv('./{}/mme_ppbg.csv'.format(save_folder), index=False)