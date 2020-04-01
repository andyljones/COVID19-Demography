import numpy as np

def p_comorbidity(country, comorbidity, warning=False):

    """
    Input:
        -country: a string input belonging to- {us, Republic of Korea, japan, Spain, italy, uk, France}
        -comorbidity: a string input belonging to- {diabetes, hypertension}
        -warning: optional, If set to True, prints out the underlying assumptions/approximations
    Returns:
        -prevalence, sampled from a prevalence array of size 100, where prevalence[i] is the prevalence rate at age between {i, i+1}
    """

    prevalence = np.zeros(101)

    ######################################  Italy #############################
    if country == 'Italy':

        if comorbidity=='diabetes':
            #from Global Burden of Disease study
            for i in range(101):
                if i <= 4:
                    prevalence[i] = 0.0001
                elif i <= 9:
                    prevalence[i] = 0.0009
                elif i <= 14:
                    prevalence[i] = 0.0024
                elif i <= 19:
                    prevalence[i] = 0.0091
                elif i <= 24:
                    prevalence[i] = 0.0264
                elif i <= 29:
                    prevalence[i] = 0.0356
                elif i <= 34:
                    prevalence[i] = 0.0392
                elif i <= 39:
                    prevalence[i] = 0.0428
                elif i <= 44:
                    prevalence[i] = 0.0489
                elif i <= 49:
                    prevalence[i] = 0.0638
                elif i <= 54:
                    prevalence[i] = 0.0893
                elif i <= 59:
                    prevalence[i] = 0.1277
                elif i <= 64:
                    prevalence[i] = 0.1783
                elif i <= 69:
                    prevalence[i] = 0.2106
                elif i <= 74:
                    prevalence[i] = 0.2407
                elif i <= 79:
                    prevalence[i] = 0.2851
                elif i <= 84:
                    prevalence[i] = 0.3348
                elif i <= 89:
                    prevalence[i] = 0.3517
                else:
                    prevalence[i] = 0.3354

        elif comorbidity=='hypertension':
            #https://www.ncbi.nlm.nih.gov/pubmed/28487768
            for i in range(101):
                if i<35:
                    prevalence[i]= 0.14*(i/35.)
                elif i<39:
                    prevalence[i]=0.14
                elif i<44:
                    prevalence[i]=0.1
                elif i<49:
                    prevalence[i]=0.16
                elif i<54:
                    prevalence[i]=0.3
                else:
                    prevalence[i]=0.34
    else:
        raise ValueError(f'{country} not supported by comorbidity sampler')

    return prevalence

def sample_joint(age, p_diabetes, p_hyp):
    #https://www-nature-com.ezp-prod1.hul.harvard.edu/articles/hr201767
    p_hyp_given_diabetes = 0.5
    p_hyp_given_not_diabetes = (p_hyp - p_hyp_given_diabetes*p_diabetes)/(1 - p_diabetes)
    diabetes_status = (np.random.rand(age.shape[0]) < p_diabetes[age]).astype(np.int)
    hyp_status = np.zeros(age.shape[0], dtype=np.int)
    hyp_status[diabetes_status == 1] = np.random.rand((diabetes_status == 1).sum()) < p_hyp_given_diabetes
    hyp_status[diabetes_status == 0] = np.random.rand((diabetes_status == 0).sum()) < p_hyp_given_not_diabetes[age[diabetes_status == 0]]
    return diabetes_status, hyp_status

def sample_joint_comorbidities(age, country='China'):
    """
    Default country is China.
    For other countries pass value for country from {us, Republic of Korea, japan, Spain, italy, uk, France}
    """

    return sample_joint(age, p_comorbidity(country, 'diabetes'), p_comorbidity(country, 'hypertension'))
