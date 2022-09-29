from utils import *
import pprint
import nltk
from naiveB import *
nltk.download('stopwords')

def naive_bayes():
	percentage_positive_instances_train = 0.1
	percentage_negative_instances_train = 0.5

	percentage_positive_instances_test  = 1
	percentage_negative_instances_test  = 1
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

	print("Number of positive training instances:", len(pos_train))
	print("Number of negative training instances:", len(neg_train))
	print("Number of positive test instances:", len(pos_test))
	print("Number of negative test instances:", len(neg_test))

	with open('vocab.txt','w') as f:
		for word in vocab:
			f.write("%s\n" % word)
	print("Vocabulary (training set):", len(vocab))

	trainer = naiveB(10)
	trainer.training(pos_train, neg_train, vocab)
	trainer.smoothing(pos_test, neg_test)
	
	
naive_bayes()
