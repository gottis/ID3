import ToyData as td
import ID3

import numpy as np
from sklearn import tree, metrics, datasets
import graphviz


def main():
    ############################### PART 1 #######################################
    digits = datasets.load_digits()
    #print(digits)

    num_examples = len(digits.data)
    num_split = int(0.7*num_examples)
    train_features = digits.data[:num_split]
    train_target = digits.target[:num_split]
    test_features = digits.data[num_split:]
    test_target = digits.target[num_split:]
    print(train_features[0])
    cart = tree.DecisionTreeClassifier()
    myTree = cart.fit(train_features, train_target)

    # https://scikit-learn.org/stable/modules/tree.html for below 3 rows of code
    dot_data = tree.export_graphviz(cart, out_file=None)
    plot = graphviz.Source(dot_data)
    plot.render("output")

    predicted = myTree.predict(test_features)
    print(metrics.classification_report(test_target, predicted))
    print(metrics.confusion_matrix(test_target, predicted))
    ############################### END OF PART 1 ##################################
    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    id3 = ID3.ID3DecisionTreeClassifier()
    my_way = []
    for i in range(len(data)):
        temp = list(data[i])
        temp.append(target[i])
        my_way.append(temp)
    #classes = (0,1,2,3,4,5,6,7,8,9)

    myTree = id3.fit(my_way, attributes, classes)
    print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(myTree, data2)
    print(predicted)
    print(metrics.classification_report(target2, predicted))
    print(metrics.confusion_matrix(target2, predicted))

if __name__ == "__main__": main()
