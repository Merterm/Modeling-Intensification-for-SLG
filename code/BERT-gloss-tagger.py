import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import Counter
import os
import ktrain
from ktrain import text

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
# MODEL_NAME = "bert-base-german-cased"
# MODEL_NAME = "bert-base-multilingual-cased"
MODEL_NAME = "bert-base-multilingual-uncased"
# MODEL_NAME = "dbmdz/bert-base-german-uncased"

def print_evaluation(gold_labels, predicted_labels):

    '''Prints accuracy, precision, recall, f1 score'''

    accuracy = accuracy_score(gold_labels, predicted_labels) * 100
    f1 = f1_score(gold_labels, predicted_labels, average = "macro") * 100
    recall = recall_score(gold_labels, predicted_labels, average = "macro") * 100
    precision = precision_score(gold_labels, predicted_labels, average = "macro") * 100
  

    a = [accuracy, precision,  recall, f1]
    for i in range (4):
        a[i] = round(a[i],2)

    return a

train_text = []
train_gloss = []
train_labels = []
train_serials = []
train_inp = []

dev_text = []
dev_gloss = []
dev_labels = []
dev_serials = []
dev_inp = []

test_text = []
test_gloss = []
test_labels = []
test_serials = []
test_inp = []


prev = ""
check = 0
max_len = 0
with open ("../data/gloss-annotation.tsv", "r") as f:
	for line in f:
		check += 1
		if check == 1:
			continue

		row = line.lower().strip().split("\t")


		try:
			split = row[0]
			serial = row[1]
			g_text = row[2]
			g_gloss_token = row[6]
			label = row[8]

			if g_text == "":
				g_text = prev
			else:
				prev = g_text
		except:
			print (check)

		if (len(g_text) > max_len):
			max_len = len(g_text)

		# print (g_text, g_gloss_token)
		if split == "train":
			train_text.append(g_text)
			train_gloss.append(g_gloss_token)
			train_labels.append(label)
			train_serials.append(serial)
			train_inp.append((g_text, g_gloss_token))

		if split == "dev":
			dev_text.append(g_text)
			dev_gloss.append(g_gloss_token)
			dev_labels.append(label)
			dev_serials.append(serial)
			dev_inp.append((g_text, g_gloss_token))

		if split == "test":
			test_text.append(g_text)
			test_gloss.append(g_gloss_token)
			test_labels.append(label)
			test_serials.append(serial)
			test_inp.append((g_text, g_gloss_token))


print (MODEL_NAME)
t = text.Transformer(MODEL_NAME, maxlen=230, class_names=list(set(train_labels)))
trn = t.preprocess_train(train_inp, train_labels)
val = t.preprocess_test(dev_inp, dev_labels)
tst = t.preprocess_test(test_inp, test_labels)
model = t.get_classifier()

learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=8) # lower bs if OOM occurs
learner.fit_onecycle(5e-5, 5)
clf = ktrain.get_predictor(learner.model, t)

dev_preds = clf.predict(dev_inp)
test_preds = clf.predict(test_inp)

print ("dev results", print_evaluation(dev_labels, dev_preds))
print ("test results", print_evaluation(test_labels, test_preds))

print (Counter(test_preds))

clf.save('gloss-tagger-model/mert-uncased')




