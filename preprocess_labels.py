import pdb
import pickle
page = open("data/ground-truth-training-20180831.txt")
text = page.read()
labels = text.split("article")

labels = labels[2:]
article_id_to_label_binary = {}
label_map = {"false": 0, "true": 1}
for label in labels:
	try:
		article_id = label.split("id=")[1].split(" ")[0].strip()
		article_id = article_id.replace("'", "")
		label = label.split("hyperpartisan=")[1].split(" ")[0]
		label = label.replace("\'", "")
		article_id_to_label_binary[article_id] = label_map[label]
	except:
		continue

pickle.dump(article_id_to_label_binary, open("labels-training-20180831-binary-dict", "wb"))
