from collections import Counter
from graphviz import Digraph
import math


class ID3DecisionTreeClassifier:

    def __init__(self, minSamplesLeaf=1, minSamplesSplit=2):

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)
        return

    # make the visualisation available
    def make_dot_data(self):
        return self.__dot

    # n är antal objekt i dataset, labels är typer av resultat
    def entropy(self, labels, n):
        ent = 0
        print(labels)
        for label in labels.keys():
            density = labels[label] / n
            ent += - density * math.log(density, 2)
            print(ent)
        return ent

    def labels(self, data, target):
        dictionary = {}
        for x in data:
            for y in x:
                if y not in dictionary.keys():
                    dictionary[y] = 1
                else:
                    dictionary[y] += 1
        return dictionary

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    # För att hitta bästa split attribut så undersöker vi entropin hos varje attribut och jämför vilken attribut som ger oss högst entropi minskning jämfört med datasettet
    def find_split_attr(self, data, remaining_attributes, target, classes):

        best_gain = None
        best_attribute = None
        dictionary = {}
        for x in range(len(data)):
            for y in data[x]:
                if (y, target[x]) not in dictionary.keys():
                    dictionary[(y, target[x])] = 1
                else:
                    dictionary[(y, target[x])] += 1
        labels = self.labels(data, target)
        ent = self.entropy(labels, len(data))
        for remaining_attribute in remaining_attributes.values():
            for att in remaining_attribute:
                entropy = 0
                for label in classes:
                    if (att, label) in dictionary.keys():
                        proportion = dictionary[(att, label)] / labels[att]
                        entropy += -proportion * math.log(proportion, 2)
                    if entropy != 0:
                        info_gain = ent - entropy/(entropy*len(data))
        return best_attribute

    def fit(self, data, target, attributes, classes):
        # Entropi av "Decision" (+ eller -) .
        #ent = self.entropy(target, classes)
        # Hitta bästa splitten
        best_attribute = self.find_split_attr(data, attributes, target, classes)

        remaining_attributes = set(attributes)
        remaining_attributes.discard(best_attribute)
        # fill in something more sensible here... root should become the output of the recursive tree creation
        root = self.new_ID3_node()
        self.add_node_to_graph(root)

        return root

    def predict(self, data, tree):
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted
