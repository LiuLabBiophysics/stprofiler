import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def RFClassication(cell_features, label_name, num=100, test_size=0.2):
    feature_array = StandardScaler().fit_transform(np.array(cell_features.drop(columns=[label_name]).dropna()))
    le_case = LabelEncoder()
    le_case.fit(cell_features[label_name])
    label_case = le_case.transform(cell_features[label_name])

    label_num = np.unique(label_case).shape[0]
    
    if(label_num==2):
        do_roc_curve = True
    else:
        do_roc_curve = False

    cms = np.zeros((num,label_num,label_num))
    ranking = np.zeros(feature_array.shape[1])

    if(do_roc_curve):
        base_fpr = np.linspace(0,1,101)
        tprs = np.zeros((num, base_fpr.shape[0]))

    for i in tqdm(range(num)):

        train_features, test_features, train_labels, test_labels = train_test_split(feature_array, label_case, test_size=test_size, random_state=42)
        
        ibrf = BalancedRandomForestClassifier(sampling_strategy='auto', replacement=True)
        ibrf.fit(train_features, train_labels)

        predictions = ibrf.predict(test_features)
        predictions_proba = ibrf.predict_proba(test_features)
        cm = confusion_matrix(test_labels, predictions, labels=ibrf.classes_, normalize='true')
        cms[i] = cm.T

        if(do_roc_curve):
            fpr, tpr, _ = roc_curve(np.array(test_labels), np.array(predictions_proba)[:,1], pos_label=1)
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs[i] = tpr
        
        importances = list(ibrf.feature_importances_)
        ranking += importances

    if(do_roc_curve):
        return cms, ranking, tprs
    else:
        return cms, ranking
    



