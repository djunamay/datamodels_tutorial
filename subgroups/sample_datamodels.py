import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from scipy import stats
from tqdm import tqdm

def compute_correct_class_margins(probs, labels):
    logit_class_1 = np.log(probs/(1-probs))
    logit_class_0 = 1-probs
    logit_class_0 = np.log(logit_class_0/(1-logit_class_0))

    margins = np.zeros_like(probs, dtype=float)
    index_class1 = labels==1
    index_class0 = labels==0

    margins[index_class1] = logit_class_1[index_class1]-logit_class_0[index_class1]
    margins[index_class0] = logit_class_0[index_class0]-logit_class_1[index_class0]
    
    return margins #the model doesnt return probabilities --> so return sign(labels) * probs as margins --> look at what they are doing in the datamodels paper and in their code




def get_samples_for_model(X, y, patient):
    """
    Return subsetted X and y matrices, specifying training sample composition and model confidences, respectively, only keeping models that were not trained on a given patient.

    Parameters:
    ----------
    X : numpy.ndarray
        Binary matrix specifying which samples (patients) were included in a given model for training.
        Shape: (N-models, N-patients)

    y : numpy.ndarray
        Matrix specifying confidence measures for a given model's (rows) prediction on a given patient (columns).
        Shape: (N-models, N-patients)

    patient : int
        The patient index of interest.

    Returns:
    -------
    y_curr : numpy.ndarray
        The model confidences for the patient of interest for models that were not trained on that patient.
        Shape: (1,N-models)

    X_curr : numpy.ndarray
        Binary matrix specifying which samples (patients) were included in a given model for training, for models that were not trained on that patient.
        Shape: (N-models, N-patients)
        
    Notes:
    ------
    Select all the models that were NOT trained on the specified patient. For those models, select all the confidence scores for that patient (i.e. how confident was a model in predicting patient class, when that patient was not included in the training set?). 
    """
    index = np.invert(X[:,patient])
    y_curr = y[index,patient]
    X_curr = X[index]
    return y_curr, X_curr


def sample_l1_lambdas(max_lambda=0.1,k=100):
    min_lambda = max_lambda/k
    lambdas = np.logspace(np.log10(min_lambda), np.log10(max_lambda), k, base=10)
    return lambdas


def fit_sample_datamodel(X_train_patient, y_train_patient, y_test_patient, X_test_patient, sample_id, lambda_id, pearson_out, mean_squared_error_out, mean_squared_error_out_train, y_test_patient_out, y_test_pred_patient_out, lambdas):
    clf = linear_model.Lasso(alpha=lambdas[lambda_id])
    clf.fit(X_train_patient, y_train_patient)
    y_test_pred_patient = clf.predict(X_test_patient)
    pearson_out[sample_id][lambda_id] = stats.pearsonr(y_test_patient, y_test_pred_patient)[0]
    mean_squared_error_out[sample_id][lambda_id] = mean_squared_error(y_test_patient, y_test_pred_patient)
    
    mean_squared_error_out_train[sample_id][lambda_id] = mean_squared_error(y_train_patient, clf.predict(X_train_patient))
    
    y_test_patient_out[sample_id][lambda_id][:len(y_test_patient)] = y_test_patient
    y_test_pred_patient_out[sample_id][lambda_id][:len(y_test_patient)] = y_test_pred_patient
    
def fit_sample_datamodel_multiple_lambdas(sample_id, X_train, y_train,X_test, y_test, lambdas, pearson_out, mean_squared_error_out, mean_squared_error_out_train, y_test_patient_out, y_test_pred_patient_out):
    
    y_train_patient, X_train_patient = get_samples_for_model(X_train, y_train, sample_id)
    y_test_patient, X_test_patient = get_samples_for_model(X_test, y_test, sample_id)

    for lambda_id in tqdm(range(len(lambdas))):
        fit_sample_datamodel(X_train_patient, y_train_patient, y_test_patient, X_test_patient, sample_id, lambda_id, pearson_out, mean_squared_error_out, mean_squared_error_out_train, y_test_patient_out, y_test_pred_patient_out, lambdas)
        
def fit_sample_datamodel_lambda_sele(sample_id, X_train, y_train,X_test, y_test, lambdas, mean_squared_error_out_test, mean_squared_error_out_train, y_test_patient_out, y_test_pred_patient_out, lambda_out, coef_out):
    
    y_train_patient, X_train_patient = get_samples_for_model(X_train, y_train, sample_id)
    y_test_patient, X_test_patient = get_samples_for_model(X_test, y_test, sample_id)
    
    current_lambda = lambdas[sample_id]
    
    clf = linear_model.Lasso(alpha=current_lambda)
    clf.fit(X_train_patient, y_train_patient)
    
    y_test_pred_patient = clf.predict(X_test_patient)
        
    mean_squared_error_out_test[sample_id] = mean_squared_error(y_test_patient, y_test_pred_patient)
    mean_squared_error_out_train[sample_id] = mean_squared_error(y_train_patient, clf.predict(X_train_patient))
    
    y_test_patient_out[sample_id][:len(y_test_patient)] = y_test_patient
    y_test_pred_patient_out[sample_id][:len(y_test_patient)] = y_test_pred_patient
    
    lambda_out[sample_id] = current_lambda
    coef_out[sample_id] = clf.coef_
