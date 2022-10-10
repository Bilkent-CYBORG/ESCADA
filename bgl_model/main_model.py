from scipy.optimize import lsq_linear
from sklearn.preprocessing import StandardScaler

import GPy
from utils.utils import *

class BGL_MODEL:
    def __init__(self, X_init, Y_init, postBG_hipohyper_range=(70,180), postBG_target_range=(70,180),
                 postBG_target_value=112.5, postBG_unit='mg/dl', variables=None, normalize_X=True, normalize_Y=True, model_gp=None):

        self.n_initdata = X_init.shape[0]
        self.normalize_X = normalize_X
        self.normalize_Y = normalize_Y 

        self.postBG_hipohyper_range = postBG_hipohyper_range
        self.postBG_target_range = postBG_target_range
        self.postBG_target_value = postBG_target_value
        self.variables = variables  
        self.units = {v['name']: v['unit'] for v in variables}
        self.units['bg_postprandial'] = postBG_unit

        self.domains = {v['name']:v['domain'] for v in variables}  
        self.org_domains = {v['name']: v['domain'] for v in variables}  
        self.scalers = [StandardScaler() if self.normalize_X else None for _ in self.variables]
        self._update_scalers(X_init)

        self.orig_X_data = X_init  
        self.X_data = scale(X_init, self.scalers) if self.normalize_X else X_init
        self.BG_data = Y_init

        self.S_p = None 

        self.model_gp = None
        self.kern_gp = None
        self.meanfunc_gp = None
        self.noise_var = None
        self._update_gp()

    def recommend(self, context, ins_calc=None, strategy=None, aux_ins=None, update=None, beta=None, lipschitz=None):

        pcontext = scale_context(context, self.variables, self.scalers)
        ins_feasible, ins_space = get_feasible_points(context=context,
                                                    domains=self.org_domains,
                                                    nsamples=self.org_domains['insulin'][1]*aux_ins + 1) 


        ins_idx = [i for i in range(len(self.variables)) if self.variables[i]['name'] == 'insulin'][0]
        ins_feasible_scaled = scale(ins_feasible, self.scalers[ins_idx]) if self.normalize_X else ins_feasible

        X = np.zeros(shape=(ins_feasible.shape[0],len(self.variables)))
        for d in range(len(self.variables)):
            dim_name = self.variables[d]['name']
            if dim_name == 'insulin':
                X[:, d] = ins_feasible_scaled
            else:
                X[:, d] = pcontext[dim_name]

        if strategy=='escada':
            ins = self._acqu_with_escada(context, ins_feasible, ins_space, ins_calc=ins_calc, update=update, beta=beta, lipschitz=lipschitz)
        elif strategy=='sts':
            ins = self._acqu_with_sts(context, ins_feasible, ins_space, ins_calc=ins_calc, aux_ins=aux_ins, update=update, beta=beta, lipschitz=lipschitz)
        elif strategy=='taco':
            ins = self._acqu_with_taco(context, ins_feasible, ins_space, beta=beta)
        elif strategy=='ts':
            ins = self._acqu_with_ts(context, ins_space, aux_ins=aux_ins, beta=beta)
        elif strategy=='gpucb':
            ins = self._acqu_with_gpucb(context, ins_feasible, ins_space, beta=beta)
        else:
            ins = -1

        return ins


    def _acqu_with_taco(self, context, ins_feasible, ins_space, beta):

        #print('+-+- TACO -+-+ ') 
        noise_std = 5
        target_bg = self.postBG_target_value

        X = np.zeros(shape=(ins_space.shape[0], len(self.variables)))
        pcontext = scale_context(context, self.variables, self.scalers)
        ins_idx = [i for i in range(len(self.variables)) if self.variables[i]['name'] == 'insulin'][0]
        for d in range(len(self.variables)):
            dim_name = self.variables[d]['name']
            if dim_name == 'insulin':
                X[:, d] = scale(ins_space, self.scalers[ins_idx])
            else:
                X[:, d] = pcontext[dim_name]

        m, v = self.model_gp.predict(X, include_likelihood=False)
        Q_z = ((m - beta*np.sqrt(v)).reshape(-1) - noise_std, (m + beta*np.sqrt(v)).reshape(-1) + noise_std)

        mu = {ins_space[i]:m[i][0] for i in range(len(ins_space))}   # Posterior Means
        l = {ins_space[i]:Q_z[0][i] for i in range(len(ins_space))}  # LCBs
        u = {ins_space[i]:Q_z[1][i] for i in range(len(ins_space))}  # UCBs
        w = {ins_space[i]:Q_z[1][i] - Q_z[0][i] for i in range(len(ins_space))}  # Confidence Widths

        candidate_doses = [ins for ins in ins_feasible if (target_bg >= l[ins] and target_bg <= u[ins])]

        if len(candidate_doses) == 0:
            #print('There is no dose whose C.I. contains postBG_target. '
            #      'Choosing the dose with the maximum uncertainty.') 
            width_dict = {cins: w[cins] for cins in ins_feasible}
            cins1 = max(width_dict, key=width_dict.get)
        else:
            dist_dict = {cins: abs(mu[cins] - target_bg) for cins in candidate_doses}
            cins2 = min(dist_dict, key=dist_dict.get)

        ins_rec = cins1 if len(candidate_doses) == 0 else cins2

        #print('Dose: {}'.format(ins_rec))
        #print('CI: ({:.2f}, {:.2f})'.format(l[ins_rec], u[ins_rec]))
        return ins_rec


    def _acqu_with_escada(self, context, ins_feasible, ins_space, ins_calc, update, beta, lipschitz):

        #print('+-+- ESCADA -+-+') 
        ins_idx = [i for i in range(len(self.variables)) if self.variables[i]['name'] == 'insulin'][0]
        var_names = [v['name'] for v in self.variables if v['name'] != 'insulin']

        noise_std = 5 
        bg_omin, bg_omax = self.postBG_target_range[0], self.postBG_target_range[1]  
        target_bg = self.postBG_target_value
        
        con = tuple(np.array([context[v] for v in var_names], dtype='int'))
        self.S_p = dict() if self.S_p is None else self.S_p

        if con not in self.S_p:
            ins_safe_discreate = min(ins_feasible, key=lambda x: abs(x - ins_calc))
            self.S_p[con] = set([ins_safe_discreate])

        X = np.zeros(shape=(ins_space.shape[0], len(self.variables)))
        pcontext = scale_context(context, self.variables, self.scalers)
        for d in range(len(self.variables)):
            dim_name = self.variables[d]['name']
            if dim_name == 'insulin':
                X[:, d] = scale(ins_space, self.scalers[ins_idx])
            else:
                X[:, d] = pcontext[dim_name]

        m, v = self.model_gp.predict(X, include_likelihood=False)
        Q_z = ((m - beta*np.sqrt(v)).reshape(-1) - noise_std, (m + beta*np.sqrt(v)).reshape(-1) + noise_std)

        mu = {ins_space[i]:m[i][0] for i in range(len(ins_space))}   # Posterior Means
        l = {ins_space[i]:Q_z[0][i] for i in range(len(ins_space))}  # LCBs
        u = {ins_space[i]:Q_z[1][i] for i in range(len(ins_space))}  # UCBs
        w = {ins_space[i]:Q_z[1][i] - Q_z[0][i] for i in range(len(ins_space))}  # Confidence Widths

        def get_one_step_closure(safe_set):
            S_p = set()
            for sins in safe_set:
                search_list = [item for item in ins_space if item not in list(safe_set)]
                for ins in search_list:
                    if (ins < sins and u[sins] + lipschitz*np.abs(ins-sins) <= bg_omax) or \
                        (ins > sins and l[sins] - lipschitz* np.abs(ins-sins) >= bg_omin):
                        S_p.add(ins)
            return S_p

        if update:
            for i in range(10):
                safe_prev = self.S_p[con]
                temp_Sp = get_one_step_closure(self.S_p[con])
                self.S_p[con] = self.S_p[con].union(temp_Sp)
                if self.S_p[con] == safe_prev:
                    break

        candidate_doses = [ins for ins in self.S_p[con] if (target_bg >= l[ins] and target_bg <= u[ins])] 

        if len(candidate_doses) == 0:
            #print('There is no safe dose whose C.I. contains postBG_target. '
            #      'Choosing the dose with the maximum uncertainty.') 
            width_dict = {cins: w[cins] for cins in list(self.S_p[con])}
            cins1 = max(width_dict, key=width_dict.get)
        else:
            dist_dict = {cins: abs(mu[cins] - target_bg) for cins in candidate_doses}
            cins2 = min(dist_dict, key=dist_dict.get)

        ins_rec = cins1 if len(candidate_doses) == 0 else cins2
   
        #print('Dose: {}'.format(ins_rec))
        #print('CI: ({:.2f}, {:.2f})'.format(l[ins_rec], u[ins_rec]))
        return ins_rec

    def _acqu_with_sts(self, context, ins_feasible, ins_space, ins_calc, aux_ins, update, beta, lipschitz):

        #print('+-+- STS -+-+') 
        ins_idx = [i for i in range(len(self.variables)) if self.variables[i]['name'] == 'insulin'][0]
        var_names = [v['name'] for v in self.variables if v['name'] != 'insulin']

        noise_std = 5 
        bg_omin, bg_omax = self.postBG_target_range[0], self.postBG_target_range[1]  
        target_bg = self.postBG_target_value
        
        con = tuple(np.array([context[v] for v in var_names], dtype='int'))
        self.S_p = dict() if self.S_p is None else self.S_p

        if con not in self.S_p:
            ins_safe_discreate = min(ins_feasible, key=lambda x: abs(x - ins_calc))
            self.S_p[con] = set([ins_safe_discreate])

        X = np.zeros(shape=(ins_space.shape[0], len(self.variables)))
        pcontext = scale_context(context, self.variables, self.scalers)
        for d in range(len(self.variables)):
            dim_name = self.variables[d]['name']
            if dim_name == 'insulin':
                X[:, d] = scale(ins_space, self.scalers[ins_idx])
            else:
                X[:, d] = pcontext[dim_name]

        m, v = self.model_gp.predict(X, include_likelihood=False)
        Q_z = ((m - beta*np.sqrt(v)).reshape(-1) - noise_std, (m + beta*np.sqrt(v)).reshape(-1) + noise_std)

        l = {ins_space[i]:Q_z[0][i] for i in range(len(ins_space))}  # LCBs
        u = {ins_space[i]:Q_z[1][i] for i in range(len(ins_space))}  # UCBs

        sample = self.model_gp.posterior_samples(X, size=1).reshape(X.shape[0], -1)
        sample_dict = {np.round(key/aux_ins, 2): dict(enumerate(sample))[key] for key in [*dict(enumerate(sample)).keys()]}
        proximity_dict = {k: np.abs(v - target_bg) for k, v in sample_dict.items()}

        def get_one_step_closure(safe_set):
            S_p = set()
            for sins in safe_set:
                search_list = [item for item in ins_space if item not in list(safe_set)]
                for ins in search_list:
                    if (ins < sins and u[sins] + lipschitz*np.abs(ins-sins) <= bg_omax) or \
                        (ins > sins and l[sins] - lipschitz* np.abs(ins-sins) >= bg_omin):
                        S_p.add(ins)
            return S_p
        
        if update:
            for i in range(10):
                safe_prev = self.S_p[con]
                temp_Sp = get_one_step_closure(self.S_p[con])
                self.S_p[con] = self.S_p[con].union(temp_Sp)

                if self.S_p[con] == safe_prev:
                    break

        filtered_samples = {key: proximity_dict[key] for key in list(self.S_p[con])}
        ins_rec = min(filtered_samples, key=filtered_samples.get)

        #print('Dose: {}'.format(ins_rec))
        #print('CI: ({:.2f}, {:.2f})'.format(l[ins_rec], u[ins_rec]))
        return ins_rec

    def _acqu_with_ts(self, context, ins_space, aux_ins, beta):
        
        #print('+-+- TS -+-+') 
        target_bg = self.postBG_target_value
        noise_std = 5
        
        X = np.zeros(shape=(ins_space.shape[0], len(self.variables)))
        ins_idx = [i for i in range(len(self.variables)) if self.variables[i]['name'] == 'insulin'][0]
        pcontext = scale_context(context, self.variables, self.scalers)
        for d in range(len(self.variables)):
            dim_name = self.variables[d]['name']
            if dim_name == 'insulin':
                X[:, d] = scale(ins_space, self.scalers[ins_idx])
            else:
                X[:, d] = pcontext[dim_name]

        m, v = self.model_gp.predict(X, include_likelihood=False)
        Q_z = ((m - beta*np.sqrt(v)).reshape(-1) - noise_std, (m + beta*np.sqrt(v)).reshape(-1) + noise_std)

        l = {ins_space[i]:Q_z[0][i] for i in range(len(ins_space))}  # LCBs
        u = {ins_space[i]:Q_z[1][i] for i in range(len(ins_space))}  # UCBs
        
        sample = self.model_gp.posterior_samples(X, size=1).reshape(X.shape[0], -1)
        sample_dict = {np.round(key/aux_ins, 2): dict(enumerate(sample))[key] for key in [*dict(enumerate(sample)).keys()]}
        proximity_dict = {k: np.abs(v - target_bg) for k, v in sample_dict.items()}
        ins_rec = min(proximity_dict, key=proximity_dict.get)

        #print('Dose: {}'.format(ins_rec))
        #print('CI: ({:.2f}, {:.2f})'.format(l[ins_rec], u[ins_rec]))

        return ins_rec

    def _acqu_with_gpucb(self, context, ins_feasible, ins_space, beta):

        #print('+-+- GP-UCB -+-+ ') 
        noise_std = 0
        target_bg = self.postBG_target_value

        X = np.zeros(shape=(ins_space.shape[0], len(self.variables)))
        pcontext = scale_context(context, self.variables, self.scalers)
        ins_idx = [i for i in range(len(self.variables)) if self.variables[i]['name'] == 'insulin'][0]
        for d in range(len(self.variables)):
            dim_name = self.variables[d]['name']
            if dim_name == 'insulin':
                X[:, d] = scale(ins_space, self.scalers[ins_idx])
            else:
                X[:, d] = pcontext[dim_name]

        m, v = self.model_gp.predict(X, include_likelihood=False)
        Q_z = ((m - beta*np.sqrt(v)).reshape(-1) - noise_std, (m + beta*np.sqrt(v)).reshape(-1) + noise_std)

        u = {ins_space[i]:Q_z[1][i] for i in range(len(ins_space))}  # UCBs
        ins_rec = max(u, key=u.get)

        #print('Dose: {}'.format(ins_rec))
        return ins_rec


    def update(self, context, insulin, postBG):
 
        if type(context) == dict and type(insulin) in {float, int, np.float16, np.float32, np.float64} and \
                type(postBG) in {float, int, np.float16, np.float32, np.float64}:
            lcontext, linsulin, lpostBG = [context], [insulin], [postBG]
        else:
            lcontext, linsulin, lpostBG = context, insulin, postBG

        pX = np.zeros((len(lcontext), len(self.variables)))
        for i in range(len(lcontext)):
            for j in range(len(self.variables)):
                var_name = self.variables[j]['name']
                if var_name != 'insulin':
                    pX[i][j] = lcontext[i][var_name]
                else:
                    pX[i][j] = linsulin[i]
        
        postBG = np.array(lpostBG)

        self.orig_X_data = np.vstack((self.orig_X_data, pX))
        self._update_scalers(self.orig_X_data)
        self.BG_data = np.vstack((self.BG_data, np.atleast_2d(postBG).T))
        self.X_data = scale(self.orig_X_data, self.scalers) if self.normalize_X else self.orig_X_data

        self._update_gp()

        return


    def _update_gp(self):

        if self.meanfunc_gp is None:

            lb = np.array([v['linearbounds'][0] for v in self.variables])
            ub = np.array([v['linearbounds'][1] for v in self.variables])

            scaler = StandardScaler().fit(self.BG_data[:self.n_initdata, :])
            x_beta = lsq_linear(self.X_data[:self.n_initdata, :],
                                scale(self.BG_data[:self.n_initdata, 0], scaler),
                                bounds=(lb, ub), method='bvls', verbose=0)

            self.meanfunc_gp = GPy.mappings.Linear(input_dim=len(self.variables), output_dim=1)
            for i in range(len(self.variables)):
                self.meanfunc_gp.A[[i]] = x_beta.x[i]
                self.meanfunc_gp.A[[i]].constrain_fixed()

        self.noise_var = 25 / np.std(self.BG_data)**2 if self.normalize_Y else 25

        self.model_gp = create_gp(X=self.X_data, y=self.BG_data, mean_func=self.meanfunc_gp, noise_var=self.noise_var,normalize_Y=self.normalize_Y)
        self.kern_gp = self.model_gp.kern

        return

    def _update_scalers(self, X):

        for i in range(len(self.scalers)):
            if self.scalers[i]:
                self.scalers[i].fit(np.atleast_2d(X[:, i]).T)

        for i in range(len(self.variables)):
            n, d, s = self.variables[i]['name'], self.variables[i]['domain'], self.scalers[i]
            self.domains[n] = (scale(d[0], s), scale(d[1], s)) if s else d
        return