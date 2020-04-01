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
    # Until then, it diverges from numpy's `choice` and so we have to keep all the randomness generation in numba 
    # while we refactor.
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

def will_shelter(age, fraction_stay_home, p):
    #whether each individual is assigned to shelter in place
    Home_real = np.zeros(int(p.n), dtype=np.bool8)
    Home_real[:] = False
    for i in range(int(p.n_ages)):
        matches = np.where(age == i)[0]
        if matches.shape[0] > 0:
            to_stay_home = choice(matches, int(fraction_stay_home[i]*matches.shape[0]), replace=False)
            Home_real[to_stay_home] = True
    return Home_real

def initial_state(initial_infected, p):
    T, n = int(p.T), int(p.n)

    s = arrdict.arrdict(
        S=np.ones((T, n), dtype=np.bool8),
        E=np.zeros((T, n), dtype=np.bool8),
        Mild=np.zeros((T, n), dtype=np.bool8),
        Documented=np.zeros((T, n), dtype=np.bool8),
        Severe=np.zeros((T, n), dtype=np.bool8),
        Critical=np.zeros((T, n), dtype=np.bool8),
        R=np.zeros((T, n), dtype=np.bool8),
        D=np.zeros((T, n), dtype=np.bool8),
        Q=np.zeros((T, n), dtype=np.bool8),)

    s.S[0, initial_infected] = False
    s.E[0, initial_infected] = True
    s.R[0] = False
    s.D[0] = False
    s.Mild[0] = False
    s.Documented[0]=False
    s.Severe[0] = False
    s.Critical[0] = False

    return s

def initial_times(initial_infected, n):
    times = arrdict.arrdict(
        time_exposed=np.full(n, -1),
        time_infected=np.zeros(n),
        time_severe=np.zeros(n),
        time_critical=np.zeros(n),
        time_documented=np.zeros(n))

    times.time_exposed[initial_infected] = 0
    return times

def initial_time_tos(initial_infected, p):
    n = int(p.n)
    time_tos = arrdict.arrdict(
        time_to_severe=np.zeros(n),
        time_to_recovery=np.zeros(n),
        time_to_critical=np.zeros(n),
        time_to_death=np.zeros(n),
        time_to_isolate=np.zeros(n),
        time_to_activation=np.zeros(n))

    #initialize values for individuals infected at the starting step
    for i in range(initial_infected.shape[0]):
        time_tos.time_to_activation[initial_infected[i]] = common.threshold_log_normal(p.time_to_activation_mean, p.time_to_activation_std)

    return time_tos

def initial_nums(initial_infected, n):
    # total number of infections caused by every individual, -1 if never become infectious
    nums = arrdict.arrdict(
        num_infected_by=np.full(n, -1),
        num_infected_by_outside=np.full(n, -1, dtype=np.int32),
        num_infected_asympt=np.full(n, -1))

    nums.num_infected_by[initial_infected] = 0
    nums.num_infected_by_outside[initial_infected] = 0
    nums.num_infected_asympt[initial_infected] = 0

    return nums

def activate(t, i, prev, curr, ts, tts, inds, deps, p):
    #exposed -> (mildly) infected
    if t - ts.time_exposed[i] == tts.time_to_activation[i]:
        curr.Mild[i] = True
        ts.time_infected[i] = t
        curr.E[i] = False
        #draw whether they will progress to severe illness
        if rand() < deps.p_mild_severe[inds.age[i], inds.diabetes[i], inds.hypertension[i]]:
            tts.time_to_severe[i] = common.threshold_exponential(p.mean_time_to_severe)
            tts.time_to_recovery[i] = np.inf
        #draw time to recovery
        else:
            tts.time_to_recovery[i] = common.threshold_exponential(p.mean_time_mild_recovery)
            tts.time_to_severe[i] = np.inf
        #draw time to isolation
        tts.time_to_isolate[i] = common.threshold_exponential(p.mean_time_to_isolate*get_isolation_factor(inds.age[i], deps.mean_time_to_isolate_factor))
        if tts.time_to_isolate[i] == 0:
            curr.Q[i] = True

def progress(t, i, prev, curr, ts, tts, inds, deps, p):
    #symptomatic individuals

    #recovery
    if t - ts.time_infected[i] == tts.time_to_recovery[i]:
        curr.R[i] = True
        curr.Mild[i] = curr.Severe[i] = curr.Critical[i] = curr.Q[i] = False
        return True

    if prev.Mild[i] and not prev.Documented[i]:
        #mild cases are documented with some probability each day
        if rand() < p.p_documented_in_mild:
            curr.Documented[i] = True
            ts.time_documented[i] = t

    #progression between infection states
    if prev.Mild[i] and t - ts.time_infected[i] == tts.time_to_severe[i]:
        curr.Mild[i] = False
        curr.Severe[i] = True
        #assume that severe cases are always documented
        if not prev.Documented[i]:
            curr.Documented[i] = True
            ts.time_documented[i] = t
        curr.Q[i] = True
        ts.time_severe[i] = t
        if rand() < deps.p_severe_critical[inds.age[i], inds.diabetes[i], inds.hypertension[i]]:
            tts.time_to_critical[i] = common.threshold_exponential(p.mean_time_to_critical)
            tts.time_to_recovery[i] = np.inf
        else:
            tts.time_to_recovery[i] = common.threshold_exponential(p.mean_time_severe_recovery) + tts.time_to_severe[i]
            tts.time_to_critical[i] = np.inf
    elif prev.Severe[i] and t - ts.time_severe[i] == tts.time_to_critical[i]:
        curr.Severe[i] = False
        curr.Critical[i] = True
        ts.time_critical[i] = t
        if rand() < deps.p_critical_death[inds.age[i], inds.diabetes[i], inds.hypertension[i]]:
            tts.time_to_death[i] = common.threshold_exponential(p.mean_time_to_death)
            tts.time_to_recovery[i] = np.inf
        else:
            tts.time_to_recovery[i] = common.threshold_exponential(p.mean_time_critical_recovery) + tts.time_to_severe[i] + tts.time_to_critical[i]
            tts.time_to_death[i] = np.inf
    #risk of mortality for critically ill patients
    elif prev.Critical[i]:
        if t - ts.time_critical[i] == tts.time_to_death[i]:
            curr.Critical[i] = False
            curr.Q[i] = False
            curr.D[i] = True

    return False

def spread(t, i, prev, curr, ts, tts, ns, Home, households, infected_by, inds, deps, p):
    #not isolated: either enter isolation or infect others
    if not prev.Q[i]:
        #isolation
        if not prev.E[i] and t - ts.time_infected[i] == tts.time_to_isolate[i]:
            curr.Q[i] = True
            return

        if prev.E[i] and t - ts.time_exposed[i] == tts.time_to_isolate[i]:
            curr.Q[i] = True
            return

        #infect within family
        for j in range(households.shape[1]):
            if households[i,j] == -1:
                break
            contact = households[i,j]
            infectiousness = inds.p_infect_household[i]
            if prev.E[i]:
                infectiousness *= p.asymptomatic_transmissibility
            if prev.S[contact] and rand() < infectiousness:
                curr.E[contact] = True
                ns.num_infected_by[contact] = 0
                ns.num_infected_by_outside[contact] = 0
                ns.num_infected_asympt[contact] = 0
                curr.S[contact] = False
                tts.time_to_isolate[contact] = common.threshold_exponential(p.mean_time_to_isolate_asympt*get_isolation_factor(inds.age[contact], deps.mean_time_to_isolate_factor))
                if tts.time_to_isolate[contact] == 0:
                    curr.Q[contact] = True
                ts.time_exposed[contact] = t
                tts.time_to_activation[contact] = common.threshold_log_normal(p.time_to_activation_mean, p.time_to_activation_std)
                ns.num_infected_by[i] += 1
                if prev.E[i]:
                    ns.num_infected_asympt[i] += 1

        #infect across families
        if not Home[i]:
            infectiousness = p.p_infect_given_contact
            #lower infectiousness for asymptomatic individuals
            if prev.E[i]:
                infectiousness *= p.asymptomatic_transmissibility
            #draw a Poisson-distributed number of contacts for each age group
            for contact_age in range(int(p.n_ages)):
                if deps.age_groups[contact_age].shape[0] == 0:
                    continue
                num_contacts = poisson(deps.contact_matrix[inds.age[i], contact_age])
                for j in range(num_contacts):
                    #if the contact becomes infected, handle bookkeeping
                    if rand() < infectiousness:
                        contact = choice(deps.age_groups[contact_age])
                        if prev.S[contact] and not Home[contact]:
                            curr.E[contact] = True
                            ns.num_infected_by[contact] = 0
                            ns.num_infected_by_outside[contact] = 0
                            ns.num_infected_asympt[contact] = 0
                            curr.S[contact] = False
                            tts.time_to_isolate[contact] = common.threshold_exponential(p.mean_time_to_isolate_asympt*get_isolation_factor(inds.age[contact], deps.mean_time_to_isolate_factor))
                            if tts.time_to_isolate[contact] == 0:
                                curr.Q[contact] = True
                            ts.time_exposed[contact] = t
                            tts.time_to_activation[contact] = common.threshold_log_normal(p.time_to_activation_mean, p.time_to_activation_std)
                            ns.num_infected_by[i] += 1
                            infected_by[i, ns.num_infected_by_outside[i]] = contact
                            ns.num_infected_by_outside[i] += 1
                            if prev.E[i]:
                                ns.num_infected_asympt[i] += 1

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
    p = params
    T = int(params['T'])
    n = int(params['n'])

    inds = arrdict.arrdict(
        age=age, 
        diabetes=diabetes, 
        hypertension=hypertension,
        p_infect_household=p_infect_household)

    deps = arrdict.arrdict(
        p_mild_severe=p_mild_severe,
        p_severe_critical=p_severe_critical,
        p_critical_death=p_critical_death,
        contact_matrix=contact_matrix,
        mean_time_to_isolate_factor=mean_time_to_isolate_factor,
        lockdown_factor_age=lockdown_factor_age,
        fraction_stay_home=fraction_stay_home,
        age_groups=age_groups)
    
    numba_seed(int(seed))

    Home_real = will_shelter(age, fraction_stay_home, p)
    Home = np.zeros(n, dtype=np.bool8) #no one shelters in place until t_stayhome_start

    initial_infected = choice(n, int(p.initial_infected_fraction*n), replace=False)
    s = initial_state(initial_infected, p)

    infected_by = np.full((n, 100), -1, dtype=np.int32)

    ns = initial_nums(initial_infected, n)
    ts = initial_times(initial_infected, n)
    tts = initial_time_tos(initial_infected, p)
    
    print('Initialized finished')
    for t in range(1, T):
        if t % 10 == 0:
            print(f'{t}/{T}')

        if t == int(p.t_lockdown):
            contact_matrix = contact_matrix/p.lockdown_factor
        if t == int(p.t_stayhome_start):
            Home = Home_real

        for k, v in s.items():
            v[t] = v[t-1]

        for i in range(n):
            prev, curr = s[t-1], s[t]
            if prev.E[i]:
                activate(t, i, prev, curr, ts, tts, inds, deps, p)
            if (prev.Mild[i] or prev.Severe[i] or prev.Critical[i]):
                recovered = progress(t, i, prev, curr, ts, tts, inds, deps, p)
                if recovered:
                    continue
            if prev.E[i] or prev.Mild[i] or prev.Severe[i] or prev.Critical[i]:
                spread(t, i, prev, curr, ts, tts, ns, Home, households, infected_by, inds, deps, p)

    return s.S, s.E, s.Mild, s.Documented, s.Severe, s.Critical, s.R, s.D, s.Q, ns.num_infected_by, ts.time_documented, tts.time_to_activation, tts.time_to_death, tts.time_to_recovery, ts.time_critical, ts.time_exposed, ns.num_infected_asympt, age, ts.time_infected, tts.time_to_severe

def run_complete_simulation(seed, country, contact_matrix, p_mild_severe, p_severe_critical, p_critical_death, mean_time_to_isolate_factor, lockdown_factor_age, p_infect_household, fraction_stay_home, params, load_population=False):
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
        params)

    S, E, Mild, Documented, Severe, Critical, R, D, Q, num_infected_by, time_documented, \
        time_to_activation, time_to_death, time_to_recovery, time_critical, time_exposed, num_infected_asympt,\
        age, time_infected, time_to_severe = result

    result = arrdict.arrdict(S=S, E=E, D=D, Mild=Mild, Severe=Severe, Critical=Critical, R=R, Q=Q, Documented=Documented)
    result['infected'] = params.n - result.S.sum(-1)

    for k in ref_result:
        np.testing.assert_allclose(ref_result[k].sum(-1), result[k].sum(-1))