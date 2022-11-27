#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/SiyeonYoo/openSW

import sys
import pandas as pd

def load_dataset(dataset_path):
	#To-Do: Implement this function
    pd.read_csv(dataset_path)

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
    dataset_df.shape[0]
    dataset_df.groupby("age").size()
    dataset_df.groupby("sex").size()
    
def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
    from sklearn.model_selection import train_test_split
    train_test_split(dataset_df.data, dataset_df.target, test_size=testset_size)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    from sklearn.tree import DecisionTreeClassifier
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(x_train, y_train)
    accuracy_score(y_test, dt_cls.predict(x_test))
    
    from sklearn.metrics import precision_score
    precision_score(y_train, y_test)
    
    from sklearn.metrics import recall_score
    recall_score(y_train, y_test)
    
def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    from sklearn.ensemble import RandomForestClassifier
    rf_cls = RandomForestClassifier()
    rf_cls.fit(x_train, y_train)
    
    from sklearn.metrics import precision_score
    precision_score(y_train, y_test)
    
    from sklearn.metrics import recall_score
    recall_score(y_train, y_test)

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    from sklearn.svm import SVC
    svm_cls = SVC()
    svm_cls.fit(x_train, y_train)
    
    from sklearn.metrics import precision_score
    precision_score(y_train, y_test)
    
    from sklearn.metrics import recall_score
    recall_score(y_train, y_test)
    
def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)