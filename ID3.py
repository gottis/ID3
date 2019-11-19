import operator
from collections import Counter
from graphviz import Digraph
from functools import reduce
import math
import numpy as np


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

    def most_common_label(self, labels):
        most_common = max(labels, key=lambda k: labels[k])
        return most_common

    def entropy(self, data, classes):
        n = len(data)
        entropy = 0
        labels = self.count_labels(data, classes)
        for label in labels:
            entropy += - labels[label] / n * math.log(labels[label] / n, 2)
        return entropy

    def count_labels(self, data, classes):
        labels = {}
        for label in classes:
            for row in data:
                if label in row:
                    if label in labels:
                        labels[label] += 1
                    else:
                        labels[label] = 1
        return labels

    def partition_data(self, data, group_att):
        partition = []
        for row in data:
            if group_att in row:
                partition.append(row)
        return partition

    def find_split_attribute(self, data, split_attribute, classes):
        max_gain = 0
        best_attribute = None
        entropy = self.entropy(data, classes)
        n = len(data)
        partitions = {}

        for attribute in split_attribute:
            avg_ent = 0
            temp = []
            for value in split_attribute[attribute]:
                partition = self.partition_data(data, value)
                temp.append(partition)
                partition_labels = self.count_labels(partition, classes)
                partition_entropy = self.entropy(partition, partition_labels)
                avg_ent += len(partition) * partition_entropy / n
            info_gain = entropy - avg_ent
            partitions[attribute] = temp
            if info_gain > max_gain:
                max_gain = info_gain
                best_attribute = attribute
        return best_attribute, partitions[best_attribute]

    def fit(self, data, attributes, classes):
        labels = self.count_labels(data, classes)
        root = self.new_ID3_node()
        if len(labels.keys()) == 1:
            root['label'] = next(iter(labels.keys()))
            return root
        if len(attributes) == 0:
            root['label'] = self.most_common_label(labels)
            root['samples'] = len(data)
            root['classCounts'] = labels
            return root
        split_attribute, partitions = self.find_split_attribute(data, attributes, classes)
        root['attribute'] = split_attribute
        root['samples'] = len(data)
        root['classCounts'] = labels
        self.add_node_to_graph(root)

        attributes_for_subtree = dict(attributes)
        del attributes_for_subtree[split_attribute]

        for partition in partitions:
            if len(partition) == 0:
                leaf = self.new_ID3_node()
                leaf['label'] = self.most_common_label(labels)
                leaf['samples'] = len(data)

                self.add_node_to_graph(leaf)
            else:
                branch = self.fit(partition, attributes_for_subtree, classes)
                self.add_node_to_graph(branch)
        return root

    def predict(self, data, tree):
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted
