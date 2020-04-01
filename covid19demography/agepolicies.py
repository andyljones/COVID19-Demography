import csv
import numpy as np
import aljpy
from . import simulator, common, arrdict
import scipy as sp
import scipy.special
from pkg_resources import resource_filename
from datetime import date

# TODO: verify if this can be changed.
N_AGES = 101

D_END = date(2020, 5, 20) # stop

# If lockdown, by how much do divide contact matrix?
LOCKDOWN_FACTOR = 2.

# How infectious are asymptomatic cases relative to symptomatic ones
# https://science.sciencemag.org/content/early/2020/03/13/science.abb3221
ASYMPTOMATIC_TRANSMISSIBILITY = 0.55

# DON'T CHANGE: we don't want p infect household to recalibrate for different policy what ifs on mean time to isolate
MEAN_TIME_TO_ISOLATE = 4.6 # DON'T CHANGE

LOAD_POPULATION = False

TUNED = aljpy.dotdict(
    Italy=aljpy.dotdict(
        # increase probability of death for all ages and comorbidities by this amount
        mortality_multiplier=4,

        # Probability of infection given contact between two individuals
        # This is currently set arbitrarily and will be calibrated to match the empirical r0
        pigc=.029,

        population=int(10e3),
        n_infected_start=5.,
        start_date=date(2020, 1, 22),
        stay_home_date=date(2020, 3, 8),
        lockdown_date=date(2022, 12, 31))) # no lockdown

AGE_GROUPS = {
    'infected_1': '0-4', 'contact_1': '0-4', 'infected_2': '5-9',
    'contact_2': '5-9', 'infected_3': '10-14', 'contact_3': '10-14',
    'infected_4': '15-19', 'contact_4': '15-19', 'infected_5': '20-24',
    'contact_5': '20-24', 'infected_6': '25-29', 'contact_6': '25-29',
    'infected_7': '30-34', 'contact_7': '30-34', 'infected_8': '35-39',
    'contact_8': '35-39', 'infected_9': '40-44', 'contact_9': '40-44',
    'infected_10': '45-49', 'contact_10': '45-49', 'infected_11': '50-54',
    'contact_11': '50-54', 'infected_12': '55-59', 'contact_12': '55-59',
    'infected_13': '60-64', 'contact_13': '60-64', 'infected_14': '65-69',
    'contact_14': '65-69', 'infected_15': '70-74', 'contact_15': '70-74',
    'infected_16': '75-79', 'contact_16': '75-79'}

def read_contact_matrix(country):
    """Create a country-specific contact matrix from stored data.

    Read a stored contact matrix based on age intervals. Return a matrix of
    expected number of contacts for each pair of raw ages. Extrapolate to age
    ranges that are not covered.

    Args:
        country (str): country name.

    Returns:
        float N_AGES x N_AGES matrix: expected number of contacts between of a person
            of age i and age j is Poisson(matrix[i][j]).
    """
    matrix = np.zeros((N_AGES, N_AGES))
    with open(resource_filename(__package__, f'contactmatrices/{country}/All_{country}.csv'), 'r') as f:
        csvraw = list(csv.reader(f))
    col_headers = csvraw[0][1:-1]
    row_headers = [row[0] for row in csvraw[1:]]
    data = np.array([row[1:-1] for row in csvraw[1:]])
    for i in range(len(row_headers)):
        for j in range(len(col_headers)):
            interval_infected = AGE_GROUPS[row_headers[i]]
            interval_infected = [int(x) for x in interval_infected.split('-')]
            interval_contact = AGE_GROUPS[col_headers[j]]
            interval_contact = [int(x) for x in interval_contact.split('-')]
            for age_infected in range(interval_infected[0], interval_infected[1]+1):
                for age_contact in range(interval_contact[0], interval_contact[1]+1):
                    matrix[age_infected, age_contact] = float(data[i][j])/(interval_contact[1] - interval_contact[0] + 1)

    # extrapolate from 79yo out to 100yo
    # start by fixing the age of the infected person and then assuming linear decrease
    # in their number of contacts of a given age, following the slope of the largest
    # pair of age brackets that doesn't contain a diagonal term (since those are anomalously high)
    for i in range(interval_infected[1]+1):
        if i < 65: # 0-65
            slope = (matrix[i, 70] - matrix[i, 75])/5
        elif i < 70: # 65-70
            slope = (matrix[i, 55] - matrix[i, 60])/5
        elif i < 75: # 70-75
            slope = (matrix[i, 60] - matrix[i, 65])/5
        else: # 75-80
            slope = (matrix[i, 65] - matrix[i, 70])/5

        start_age = 79
        if i >= 75:
            start_age = 70
        for j in range(interval_contact[1]+1, N_AGES):
            matrix[i, j] = matrix[i, start_age] - slope*(j - start_age)
            if matrix[i, j] < 0:
                matrix[i, j] = 0

    # fix diagonal terms
    for i in range(interval_infected[1]+1, N_AGES):
        matrix[i] = matrix[interval_infected[1]]
    for i in range(int((100-80)/5)):
        age = 80 + i*5
        matrix[age:age+5, age:age+5] = matrix[79, 79]
        matrix[age:age+5, 75:80] = matrix[75, 70]
    matrix[100, 95:] = matrix[79, 79]
    matrix[95:, 100] = matrix[79, 79]

    return matrix

def transition_probabilities(mortality_multiplier):
    """2b. Construct transition probabilities between disease severities
    There are three disease states: mild, severe and critical.
    - Mild represents sub-hospitalization.
    - Severe is hospitalization.
    - Critical is ICU.

    The key results of this section are:
    - p_mild_severe: N_AGES x 2 x 2 matrix. For each age and comorbidity state
        (length two bool vector indicating whether the individual has diabetes and/or
        hypertension), what is the probability of the individual transitioning from
        the mild to severe state.
    - p_severe_critical, p_critical_death are the same for the other state transitions.

    All of these probabilities are proportional to the base progression rate
    for an (age, diabetes, hypertension) state which is stored in p_death_target
    and estimated via logistic regression.
    """

    # N_AGES vector: The probability of transitioning from the mild to
    #     severe state for a patient of age i is p_mild_severe_cdc[i]. We will match
    #     these overall probabilities.

    # Source: https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm?s_cid=mm6912e2_w#T1_down
    # Using the lower bounds for probability of hospitalization, since that's more
    # consistent with frequency of severe infection reported in
    # https://www.nejm.org/doi/full/10.1056/NEJMoa2002032 (at a lower level of age granularity).
    p_mild_severe_cdc = np.zeros(N_AGES)
    p_mild_severe_cdc[0:20] = 0.016
    p_mild_severe_cdc[20:45] = 0.143
    p_mild_severe_cdc[45:55] = 0.212
    p_mild_severe_cdc[55:65] = 0.205
    p_mild_severe_cdc[65:75] = 0.286
    p_mild_severe_cdc[75:85] = 0.305
    p_mild_severe_cdc[85:] = 0.313

    # overall probability of progression from critical to severe
    # https://www.ecdc.europa.eu/sites/default/files/documents/RRA-sixth-update-Outbreak-of-novel-coronavirus-disease-2019-COVID-19.pdf
    # taking midpoint of the intervals
    overall_p_severe_critical = (0.15 + 0.2) / 2

    # overall mortality, which is set separately, but rather how many individuals
    # end up in critical state. 0.49 is from
    # http://weekly.chinacdc.cn/en/article/id/e53946e2-c6c4-41e9-9a9b-fea8db1a8f51
    overall_p_critical_death = 0.49

    # go back to using CDC hospitalization rates as mild->severe
    severe_critical_multiplier = overall_p_severe_critical / p_mild_severe_cdc
    critical_death_multiplier = overall_p_critical_death / p_mild_severe_cdc

    # get the overall CFR for each age/comorbidity combination by running the logistic model
    """
    Mortality model. We fit a logistic regression to estimate p_mild_death from
    (age, diabetes, hypertension) to match the marginal mortality rates from TODO.
    The results of the logistic regression are used to set the disease severity
    transition probabilities.
    """
    c_age = np.loadtxt(resource_filename(__package__, 'comorbidities/c_age.txt'), delimiter=',').mean(axis=0)
    """float vector: Logistic regression weights for each age bracket."""
    c_diabetes = np.loadtxt(resource_filename(__package__, 'comorbidities/c_diabetes.txt'), delimiter=',').mean(axis=0)
    """float: Logistic regression weight for diabetes."""
    c_hyper = np.loadtxt(resource_filename(__package__, 'comorbidities/c_hypertension.txt'), delimiter=',').mean(axis=0)
    """float: Logistic regression weight for hypertension."""
    intervals = np.loadtxt(resource_filename(__package__, 'comorbidities/comorbidity_age_intervals.txt'), delimiter=',')

    def age_to_interval(i):
        """Return the corresponding comorbidity age interval for a specific age.

        Args:
            i (int): age.

        Returns:
            int: index of interval containing i in intervals.
        """
        for idx, a in enumerate(intervals):
            if i >= a[0] and i < a[1]:
                return idx
        return idx

    p_death_target = np.zeros((N_AGES, 2, 2))
    for i in range(N_AGES):
        for diabetes_state in [0,1]:
            for hyper_state in [0,1]:
                if i < intervals[0][0]:
                    p_death_target[i, diabetes_state, hyper_state] = 0
                else:
                    p_death_target[i, diabetes_state, hyper_state] = sp.special.expit(
                        c_age[age_to_interval(i)] + diabetes_state * c_diabetes +
                        hyper_state * c_hyper)

    # p_death_target *= params['mortality_multiplier']
    # p_death_target[p_death_target > 1] = 1


    #calibrate the probability of the severe -> critical transition to match the
    #overall CFR for each age/comorbidity combination
    #age group, diabetes (0/1), hypertension (0/1)
    progression_rate = np.zeros((N_AGES, 2, 2))
    p_mild_severe = np.zeros((N_AGES, 2, 2))
    """float N_AGES x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from mild to severe state."""
    p_severe_critical = np.zeros((N_AGES, 2, 2))
    """float N_AGES x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from severe to critical state."""
    p_critical_death = np.zeros((N_AGES, 2, 2))
    """float N_AGES x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from critical to dead state."""

    for i in range(N_AGES):
        for diabetes_state in [0,1]:
            for hyper_state in [0,1]:
                progression_rate[i, diabetes_state, hyper_state] = (p_death_target[i, diabetes_state, hyper_state]
                                                                    / (severe_critical_multiplier[i]
                                                                    * critical_death_multiplier[i])) ** (1./3)
                p_mild_severe[i, diabetes_state, hyper_state] = progression_rate[i, diabetes_state, hyper_state]
                p_severe_critical[i, diabetes_state, hyper_state] = severe_critical_multiplier[i]*progression_rate[i, diabetes_state, hyper_state]
                p_critical_death[i, diabetes_state, hyper_state] = critical_death_multiplier[i]*progression_rate[i, diabetes_state, hyper_state]
    #no critical cases under 20 (CDC)
    p_critical_death[:20] = 0
    p_severe_critical[:20] = 0
    #for now, just cap 80+yos with diabetes and hypertension
    p_critical_death[p_critical_death > 1] = 1

    p_mild_severe *= mortality_multiplier**(1/3)
    p_severe_critical *= mortality_multiplier**(1/3)
    p_critical_death *= mortality_multiplier**(1/3)
    p_mild_severe[p_mild_severe > 1] = 1
    p_severe_critical[p_severe_critical > 1] = 1
    p_critical_death[p_critical_death > 1] = 1

    return aljpy.dotdict(
        p_mild_severe=p_mild_severe,
        p_severe_critical=p_severe_critical,
        p_critical_death=p_critical_death,
    )

def lockdown_factor(factor):
    return np.array(((0, 14, factor), (15, 24, factor), (25, 39, factor), (40, 69, factor), (70, 100, factor)))

def mtti_factor():
    # TODO: Find documented probabilities, age distribution or mean_time
    return np.array(((0, 14, 1), (14, 24, 1), (25, 39, 1), (40, 69, 1), (70, 100, 1)))

def mean_times(mtti):
    params = aljpy.dotdict()

    params['mean_time_to_isolate'] = mtti

    #for now, the time for all of these events will be exponentially distributed
    #from https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf
    params['mean_time_to_severe'] = 7.
    params['mean_time_mild_recovery'] = 14.

    #guessing based on time to mechanical ventilation as 14.5 days from
    #https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext
    #and subtracting off the 7 to get to critical. This also matches mortality risk
    #starting at 2 weeks in the WHO report
    params['mean_time_to_critical'] = 7.5

    #WHO gives 3-6 week interval for severe and critical combined
    #using 4 weeks as mean for severe and 5 weeks as mean for critical
    params['mean_time_severe_recovery'] = 28. - params['mean_time_to_severe']
    params['mean_time_critical_recovery'] = 35. - params['mean_time_to_severe'] - params['mean_time_to_critical'] 
    #mean_time_severe_recovery = mean_time_critical_recovery = 21

    #mean_time_to_death = 35 #taking the midpoint of the 2-8 week interval
    #update: use 35 - mean time to severe - mean time to critical as the excess time
    #to death after reaching critical
    #update: use 18.5 days as median time onset to death from
    #https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext
    params['mean_time_to_death'] = 18.5 - params['mean_time_to_severe'] - params['mean_time_to_critical'] 
    #mean_time_to_death = 1 #this will now be critical -> death

    #probability of exposed individual becoming infected each time step
    #set based on https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported
    params['time_to_activation_mean'] = 1.621
    params['time_to_activation_std'] = 0.418

    params['mean_time_to_isolate_asympt'] = np.inf

    return params

def contact_tracing(start_date):
    """Whether contact tracing happens, and if so the probability of successfully 
    identifying each within and between household infected individual
    """        
    d_tracing_start = date(2020, 2, 10)
    return aljpy.dotdict(
        contact_tracing=float(False),
        p_trace_outside=1.0,
        p_trace_household=0.75,
        t_tracing_start=float((d_tracing_start - start_date).days))

def assemble_kwargs(frac_stay_home, mtti, country='Italy', seed=0):
    tuned = TUNED[country]

    times = mean_times(mtti)
    p_infect_household = common.get_p_infect_household(int(tuned.population), MEAN_TIME_TO_ISOLATE, 
                                    times.time_to_activation_mean, times.time_to_activation_std, ASYMPTOMATIC_TRANSMISSIBILITY)

    params = aljpy.dotdict(
        n=float(tuned.population),
        n_ages=float(N_AGES),
        seed=float(seed),
        T=float((D_END - tuned.start_date).days + 1),
        initial_infected_fraction=tuned.n_infected_start/tuned.population,
        t_lockdown=float((tuned.lockdown_date - tuned.start_date).days),
        lockdown_factor=LOCKDOWN_FACTOR,
        t_stayhome_start=float((tuned.stay_home_date - tuned.start_date).days),
        p_documented_in_mild=0.,
        p_infect_given_contact=tuned.pigc,
        asymptomatic_transmissibility=ASYMPTOMATIC_TRANSMISSIBILITY,
        mortality_multiplier=tuned.mortality_multiplier,
        **times,
        **contact_tracing(tuned.start_date)
    )

    return aljpy.dotdict(
        seed=seed,
        country=country,
        contact_matrix=read_contact_matrix(country),
        **transition_probabilities(TUNED[country].mortality_multiplier),
        mean_time_to_isolate_factor=mtti_factor(),
        lockdown_factor_age=lockdown_factor(LOCKDOWN_FACTOR),
        p_infect_household=p_infect_household,
        fraction_stay_home=np.asarray(frac_stay_home),
        params=params,
        load_population=LOAD_POPULATION)


def simulate(kwargs):
    # Seed is being set directly before the sim here, rather than in advance of 
    # get_p_infect_household (which has now been derandomized)
    np.random.seed(kwargs.seed)

    S, E, Mild, Documented, Severe, Critical, R, D, Q, num_infected_by, time_documented, \
        time_to_activation, time_to_death, time_to_recovery, time_critical, time_exposed, num_infected_asympt,\
        age, time_infected, time_to_severe = simulator.run_complete_simulation(**kwargs)
    
    per_time = arrdict.arrdict(S=S, E=E, D=D, mild=Mild, severe=Severe, critical=Critical, R=R, Q=Q, documented=Documented).sum(1)
    per_time['infected'] = kwargs.params.n - per_time.S

    return arrdict.arrdict(
        r0_total=num_infected_by[(0 < time_exposed) & (time_exposed <= 20)].mean(),
        per_time=per_time,
    )

def uniform_stay_at_home(frac):
    return np.full(N_AGES, frac)

def per_group_stay_at_home(fracs):
    age_ranges = [(0,14), (15,29), (30,49), (50,69), (70,100)]
    frac_stay_home = np.zeros(N_AGES)
    for (l, u), frac in zip(age_ranges, fracs):
        frac_stay_home[l:u+1] = frac 
    return frac_stay_home

def run(frac, repeats=5):
    results = []
    for i in range(repeats):
        params = assemble_kwargs(frac, 4.6, seed=1+i)
        result = simulate(params)
        results.append(result.per_time)
    results = arrdict.stack(results, -1)

    return results

def validate():
    from pathlib import Path
    import pickle

    # Requires derandomizing get_p_infect_household
    ref_kwargs = pickle.loads(Path('output/ref-kwargs.pickle').read_bytes())
    ref_result = pickle.loads(Path('output/ref-results.pickle').read_bytes())

    frac = uniform_stay_at_home(1.)
    kwargs = assemble_kwargs(frac, 4.6, seed=1)
    result = simulate(kwargs)

    for k in ref_kwargs:
        actual = kwargs[k]
        expected = ref_kwargs[k]
        
        if isinstance(expected, (float, str, bool)):
            assert expected == actual
        elif isinstance(expected, np.ndarray):
            if not np.allclose(expected, actual):
                print(k)
                break
        elif isinstance(expected, dict):
            for kk in expected:
                assert expected[kk] == actual[kk]
        else:
            print(type(expected))
            break

    actual = result.per_time.D
    expected = ref_result['D'].sum(-1)

    np.testing.assert_allclose(actual, expected)