import numpy as np
import numba
from numba import jit
from . import comorbiditysampler, householdsampler, common, arrdict
import pickle
import aljpy

LOAD_POPULATION = True

def as_numba_dict(d):
    nd = numba.typed.Dict()
    for k, v in d.items():
        nd[k] = v
    return nd

@aljpy.autocache()
def sample_population(n, country, n_ages):
    if country == "Italy":
        households, age = householdsampler.sample_households_italy(n)      
    else:
        raise ValueError(f'{country} not supported by the household sampler')
    age_groups = tuple([np.where(age == i)[0] for i in range(0, n_ages)])
    diabetes, hypertension = comorbiditysampler.sample_joint_comorbidities(age, country)

    return (age, households, diabetes, hypertension, age_groups)

@jit(nopython=True)
def get_isolation_factor(age, mean_time_to_isolate_factor):
    for i in range(len(mean_time_to_isolate_factor)):
        if age >= mean_time_to_isolate_factor[i, 0] and age <= mean_time_to_isolate_factor[i, 1]:
            return mean_time_to_isolate_factor[i, 2]
    return 1

@jit(nopython=True)
def get_lockdown_factor_age(age, lockdown_factor_age):
    for i in range(len(lockdown_factor_age)):
        if age >= lockdown_factor_age[i, 0] and age <= lockdown_factor_age[i, 1]:
            return lockdown_factor_age[i, 2]
    return 1

@jit(nopython=True)
def choice(n, k=None, replace=None):
    # np.random.choice's seed will be fixed in the next version of numba: https://github.com/numba/numba/issues/3249
    # Until then, it diverges from numpy's `choice`.
    return np.random.choice(n, k, replace=replace)

@jit(nopython=True)
def rand():
    return np.random.rand()

@jit(nopython=True)
def poisson(mu):
    return np.random.poisson(mu)

@jit(nopython=True)
def numba_seed(seed):
    return np.random.seed(seed)

# @jit(nopython=True)
def run_model(seed, households, age, age_groups, diabetes, hypertension, contact_matrix, p_mild_severe, p_severe_critical, p_critical_death, mean_time_to_isolate_factor, lockdown_factor_age, p_infect_household, fraction_stay_home, params):
    print('run_model')
    """Run the SEIR model to completion.

    Args:
        seed (int): Random seed.
        households (int n x max_household_size matrix): Household structure (adjacency list format, each row terminated with -1s)
        age (int vector of length n): Age of each individual.
        diabetes (bool vector of length n): Diabetes state of each individual.
        hypertension (bool vector of length n): Hypertension state of each
            individual.
        contact_matrix (float matrix n_ages x n_ages): expected number of daily contacts between each pair of age groups
        p_mild_severe (float matrix n_ages x 2 x 2): probability of mild->severe transition for each age/diabetes/hypertension status
        p_severe_critical (float matrix n_ages x 2 x 2): as above but for severe->critical
        p_critical_death (float matrix n_ages x 2 x 2): as above but for critical->death
        mean_time_to_isolate_factor (float vector n_ages): scaling applied to the mean time to isolate for mild cases per age group
        lockdown_factor_age (float vector n_ages): per-age reduction in contact during lockdown. Not currently used.
        p_infect_household (float vector n): probability for each individual to infect each household member per day.
        fraction_stay_home (float vector n_ages): fraction of each age group assigned to shelter in place
        params: dict with remaining scalar parameters

    Returns:
        S (bool T x n matrix): Matrix where S[i][j] represents
            whether individual i was in the Susceptible state at time j.
        E (bool T x n matrix): same for Exposed state.
        Mild (bool T x n matrix): same for Mild state.
        Severe (bool T x n matrix): same for Severe state.
        Critical (bool T x n matrix): same for Critical state.
        R (bool T x n matrix): same for Recovered state.
        D (bool T x n matrix): same for Dead state.
        Q (bool T x n matrix): same for Quarantined state.
        num_infected_by (n vector): num_infected_by[i] is the number of individuals
            infected by individual i. -1 if they never became infectious
        time_documented (n vector): time step when each individual became a documented case, 0 if never documented
        time_to_activation (n vector): incubation time drawn for this individual, 0 if no time drawn
        time_to_death (n vector):  time-to-event for critical -> death for this individual, 0 if no time drawn
        time_to_recovery (n vector): time-to-event for infectious -> recovery for this individual, 0 if no time drawn
        time_critical (n vector): time step that this individual entered the critical state, 0 if they never entered
        time_exposed (n vector): time step that this individual became asymptomatically infected, -1 if this never happend
        num_infected_asympt (n vector): number of others infected by this individual while asymptomatic, -1 if never became infectious
        age (n vector): age of each individual
        time_infected (n vector): time step that this individual became mildly infectious, 0 if they never became infectious
        time_to_severe (n vector): time-to-event between mild and severe cases for this individual, 0 if never drawn
    """
    time_to_activation_mean = params['time_to_activation_mean']
    time_to_activation_std = params['time_to_activation_std']
    mean_time_to_death = params['mean_time_to_death']
    mean_time_critical_recovery = params['mean_time_critical_recovery']
    mean_time_severe_recovery = params['mean_time_severe_recovery']
    mean_time_to_severe = params['mean_time_to_severe']
    mean_time_mild_recovery = params['mean_time_mild_recovery']
    mean_time_to_critical = params['mean_time_to_critical']
    p_documented_in_mild = params['p_documented_in_mild']
    mean_time_to_isolate_asympt = params['mean_time_to_isolate_asympt']
    asymptomatic_transmissibility = params['asymptomatic_transmissibility']
    p_infect_given_contact = params['p_infect_given_contact']
    T = int(params['T'])
    initial_infected_fraction = params['initial_infected_fraction']
    t_lockdown = int(params['t_lockdown']) 
    lockdown_factor = params['lockdown_factor']
    mean_time_to_isolate = params['mean_time_to_isolate']
    n = int(params['n'])
    n_ages = int(params['n_ages'])
    contact_tracing = bool(params['contact_tracing'])
    t_tracing_start = int(params['t_tracing_start'])
    t_stayinghome_start = int(params['t_stayhome_start'])
    
    numba_seed(int(seed))
    max_household_size = households.shape[1]
    S = np.zeros((T, n), dtype=np.bool8)
    E = np.zeros((T, n), dtype=np.bool8)
    Mild = np.zeros((T, n), dtype=np.bool8)
    Documented = np.zeros((T, n), dtype=np.bool8)
    Severe = np.zeros((T, n), dtype=np.bool8)
    Critical = np.zeros((T, n), dtype=np.bool8)
    R = np.zeros((T, n), dtype=np.bool8)
    D = np.zeros((T, n), dtype=np.bool8)
    Q = np.zeros((T, n), dtype=np.bool8)
    traced = np.zeros((n), dtype=np.bool8)
    #whether each individual is assigned to shelter in place
    Home_real = np.zeros(n, dtype=np.bool8)
    Home_real[:] = False
    for i in range(n_ages):
        matches = np.where(age == i)[0]
        if matches.shape[0] > 0:
            to_stay_home = choice(matches, int(fraction_stay_home[i]*matches.shape[0]), replace=False)
            Home_real[to_stay_home] = True
    #no one shelters in place until t_stayhome_start
    dummy_Home = np.zeros(n, dtype=np.bool8)
    dummy_Home[:] = False
    Home = dummy_Home
    initial_infected = choice(n, int(initial_infected_fraction*n), replace=False)
    S[0] = True
    E[0] = False
    R[0] = False
    D[0] = False
    Mild[0] = False
    Documented[0]=False
    Severe[0] = False
    Critical[0] = False

    infected_by = np.zeros((n, 100), dtype=np.int32)
    infected_by[:] = -1
    
    time_exposed = np.zeros(n)
    time_infected = np.zeros(n)
    time_severe = np.zeros(n)
    time_critical = np.zeros(n)
    time_documented=np.zeros(n)
    time_exposed[:] = -1
    #total number of infections caused by every individual, -1 if never become infectious
    num_infected_by = np.zeros(n)
    num_infected_by_outside = np.zeros(n, dtype=np.int32)
    num_infected_asympt = np.zeros(n)
    num_infected_by[:] = -1
    num_infected_by_outside[:] = -1
    num_infected_asympt[:] = -1
    time_to_severe = np.zeros(n)
    time_to_recovery = np.zeros(n)
    time_to_critical = np.zeros(n)
    time_to_death = np.zeros(n)
    time_to_isolate = np.zeros(n)
    time_to_activation = np.zeros(n)
    #initialize values for individuals infected at the starting step
    for i in range(initial_infected.shape[0]):
        E[0, initial_infected[i]] = True
        S[0, initial_infected[i]] = False
        time_exposed[initial_infected[i]] = 0
        num_infected_by[initial_infected[i]] = 0
        num_infected_by_outside[initial_infected[i]] = 0
        num_infected_asympt[initial_infected[i]] = 0
        time_to_activation[initial_infected[i]] = common.threshold_log_normal(time_to_activation_mean, time_to_activation_std)
    
    print('Initialized finished')
    print('mean_time_to_isolate',mean_time_to_isolate)
    for t in range(1, T):
        if t % 10 == 0:
            print(t,"/",T)
        if t == t_lockdown:
            # implements a reduction in contact per age group (instead of a single factor)
            # applied to the whole population. Currently not used.
            # for i in range(contact_matrix.shape[0]):
            #     for j in range(contact_matrix.shape[1]):
            #         contact_matrix[i,j] = contact_matrix[i,j]/(get_lockdown_factor_age(i,lockdown_factor_age)*get_lockdown_factor_age(j,lockdown_factor_age))
            contact_matrix = contact_matrix/lockdown_factor
        if t == t_stayinghome_start:
            Home = Home_real
        S[t] = S[t-1]
        E[t] = E[t-1]
        Mild[t] = Mild[t-1]
        Documented[t]=Documented[t-1]
        Severe[t] = Severe[t-1]
        Critical[t] = Critical[t-1]
        R[t] = R[t-1]
        D[t] = D[t-1]
        Q[t] = Q[t-1]
        for i in range(n):
            #exposed -> (mildly) infected
            if E[t-1, i]:
                if t - time_exposed[i] == time_to_activation[i]:
                    Mild[t, i] = True
                    time_infected[i] = t
                    E[t, i] = False
                    #draw whether they will progress to severe illness
                    if rand() < p_mild_severe[age[i], diabetes[i], hypertension[i]]:
                        time_to_severe[i] = common.threshold_exponential(mean_time_to_severe)
                        time_to_recovery[i] = np.inf
                    #draw time to recovery
                    else:
                        time_to_recovery[i] = common.threshold_exponential(mean_time_mild_recovery)
                        time_to_severe[i] = np.inf
                    #draw time to isolation
                    time_to_isolate[i] = common.threshold_exponential(mean_time_to_isolate*get_isolation_factor(age[i], mean_time_to_isolate_factor))
                    if time_to_isolate[i] == 0:
                        Q[t, i] = True
            #symptomatic individuals
            if (Mild[t-1, i] or Severe[t-1, i] or Critical[t-1, i]):
                #recovery
                if t - time_infected[i] == time_to_recovery[i]:
                    R[t, i] = True
                    Mild[t, i] = Severe[t, i] = Critical[t, i] = Q[t, i] = False
                    continue
                if Mild[t-1, i] and not Documented[t-1, i]:
                    #mild cases are documented with some probability each day
                    if rand() < p_documented_in_mild:
                        Documented[t, i] = True
                        time_documented[i] = t
                        traced[i] = True
                #progression between infection states
                if Mild[t-1, i] and t - time_infected[i] == time_to_severe[i]:
                    Mild[t, i] = False
                    Severe[t, i] = True
                    #assume that severe cases are always documented
                    if not Documented[t-1, i]:
                        Documented[t, i] = True
                        time_documented[i] = t
                        traced[i] = True
                    Q[t, i] = True
                    time_severe[i] = t
                    if rand() < p_severe_critical[age[i], diabetes[i], hypertension[i]]:
                        time_to_critical[i] = common.threshold_exponential(mean_time_to_critical)
                        time_to_recovery[i] = np.inf
                    else:
                        time_to_recovery[i] = common.threshold_exponential(mean_time_severe_recovery) + time_to_severe[i]
                        time_to_critical[i] = np.inf
                elif Severe[t-1, i] and t - time_severe[i] == time_to_critical[i]:
                    Severe[t, i] = False
                    Critical[t, i] = True
                    time_critical[i] = t
                    if rand() < p_critical_death[age[i], diabetes[i], hypertension[i]]:
                        time_to_death[i] = common.threshold_exponential(mean_time_to_death)
                        time_to_recovery[i] = np.inf
                    else:
                        time_to_recovery[i] = common.threshold_exponential(mean_time_critical_recovery) + time_to_severe[i] + time_to_critical[i]
                        time_to_death[i] = np.inf
                #risk of mortality for critically ill patients
                elif Critical[t-1, i]:
                    if t - time_critical[i] == time_to_death[i]:
                        Critical[t, i] = False
                        Q[t, i] = False
                        D[t, i] = True
            if E[t-1, i] or Mild[t-1, i] or Severe[t-1, i] or Critical[t-1, i]:
                #not isolated: either enter isolation or infect others
                if not Q[t-1, i]:
                    #isolation
                    if not E[t-1, i] and t - time_infected[i] == time_to_isolate[i]:
                        Q[t, i] = True
                        continue
                    if E[t-1, i] and t - time_exposed[i] == time_to_isolate[i]:
                        Q[t, i] = True
                        continue
                    #infect within family
                    for j in range(max_household_size):
                        if households[i,j] == -1:
                            break
                        contact = households[i,j]
                        infectiousness = p_infect_household[i]
                        if E[t-1, i]:
                            infectiousness *= asymptomatic_transmissibility
                        if S[t-1, contact] and rand() < infectiousness:
                            E[t, contact] = True
                            num_infected_by[contact] = 0
                            num_infected_by_outside[contact] = 0
                            num_infected_asympt[contact] = 0
                            S[t, contact] = False
                            time_to_isolate[contact] = common.threshold_exponential(mean_time_to_isolate_asympt*get_isolation_factor(age[contact], mean_time_to_isolate_factor))
                            if time_to_isolate[contact] == 0:
                                Q[t, contact] = True
                            time_exposed[contact] = t
                            time_to_activation[contact] = common.threshold_log_normal(time_to_activation_mean, time_to_activation_std)
                            num_infected_by[i] += 1
                            if E[t-1, i]:
                                num_infected_asympt[i] += 1
                    #infect across families
                    if not Home[i]:
                        infectiousness = p_infect_given_contact
                        #lower infectiousness for asymptomatic individuals
                        if E[t-1, i]:
                            infectiousness *= asymptomatic_transmissibility
                        #draw a Poisson-distributed number of contacts for each age group
                        for contact_age in range(n_ages):
                            if age_groups[contact_age].shape[0] == 0:
                                continue
                            num_contacts = poisson(contact_matrix[age[i], contact_age])
                            for j in range(num_contacts):
                                #if the contact becomes infected, handle bookkeeping
                                if rand() < infectiousness:
                                    contact = choice(age_groups[contact_age])
                                    if S[t-1, contact] and not Home[contact]:
                                        E[t, contact] = True
                                        num_infected_by[contact] = 0
                                        num_infected_by_outside[contact] = 0
                                        num_infected_asympt[contact] = 0
                                        S[t, contact] = False
                                        time_to_isolate[contact] = common.threshold_exponential(mean_time_to_isolate_asympt*get_isolation_factor(age[contact], mean_time_to_isolate_factor))
                                        if time_to_isolate[contact] == 0:
                                            Q[t, contact] = True
                                        time_exposed[contact] = t
                                        time_to_activation[contact] = common.threshold_log_normal(time_to_activation_mean, time_to_activation_std)
                                        num_infected_by[i] += 1
                                        infected_by[i, num_infected_by_outside[i]] = contact
                                        num_infected_by_outside[i] += 1
                                        if E[t-1, i]:
                                            num_infected_asympt[i] += 1

    return S, E, Mild, Documented, Severe, Critical, R, D, Q, num_infected_by,time_documented, time_to_activation, time_to_death, time_to_recovery, time_critical, time_exposed, num_infected_asympt, age, time_infected, time_to_severe

def run_complete_simulation(seed, country, contact_matrix, p_mild_severe, p_severe_critical, p_critical_death, mean_time_to_isolate_factor, lockdown_factor_age, p_infect_household, fraction_stay_home, params, load_population=False):
    '''
    Runs simulation with given parameters (see run_simulation for details).
    If load_population is true, reads in the simulated population for the given country and size.
    Otherwise, creates and saves a new population.
    '''
    np.random.seed(seed)
    age, households, diabetes, hypertension, age_groups = sample_population(params.n, country, params.n_ages)

    print('starting simulation')
    return run_model(seed, households, age, age_groups, diabetes, hypertension, contact_matrix, p_mild_severe, p_severe_critical, p_critical_death, mean_time_to_isolate_factor, lockdown_factor_age, p_infect_household, fraction_stay_home, as_numba_dict(params))

def validate():
    from pathlib import Path
    import pickle

    ref_kwargs = aljpy.dotdict(pickle.loads(Path('output/ref-kwargs.pickle').read_bytes()))
    ref_result = aljpy.dotdict(pickle.loads(Path('output/ref-results.pickle').read_bytes()))
    params = aljpy.dotdict(ref_kwargs.params)

    np.random.seed(int(ref_kwargs.seed))
    age, households, diabetes, hypertension, age_groups = sample_population(int(params.n), ref_kwargs.country, int(params.n_ages))

    print('starting simulation')
    result = run_model(
        ref_kwargs.seed, households, age, age_groups, diabetes, hypertension, 
        ref_kwargs.contact_matrix, ref_kwargs.p_mild_severe, ref_kwargs.p_severe_critical, ref_kwargs.p_critical_death,
        ref_kwargs.mean_time_to_isolate_factor, ref_kwargs.lockdown_factor_age, ref_kwargs.p_infect_household, ref_kwargs.fraction_stay_home,
        as_numba_dict(params))

    S, E, Mild, Documented, Severe, Critical, R, D, Q, num_infected_by, time_documented, \
        time_to_activation, time_to_death, time_to_recovery, time_critical, time_exposed, num_infected_asympt,\
        age, time_infected, time_to_severe = result
    
    result = arrdict.arrdict(S=S, E=E, D=D, mild=Mild, severe=Severe, critical=Critical, R=R, Q=Q, documented=Documented)
    result['infected'] = params.n - result.S

    np.testing.assert_allclose(ref_result.D, result.D)