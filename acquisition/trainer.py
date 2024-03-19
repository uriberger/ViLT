from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

class Trainer:
    def __init__(self, classifier, config, train_data, test_data):
        self.classifier = classifier
        self.config = config
        self.train_data = train_data
        self.test_data = test_data

class NeuralTrainer(Trainer):
    def __init__(self, model, train_data, test_data, config):
        super(NeuralTrainer, self).__init__(model, train_data, test_data, config)

    def train(self):
        dataloader = DataLoader(self.train_data, batch_size=64, shuffle=True)
        criterion = F.cross_entropy
        optimizer = optim.Adam(self.classifier.parameters())
        checkpoint_len = 100

        for epoch_ind in range(10):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                inputs, labels = data
                optimizer.zero_grad()

                outputs = self.classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % checkpoint_len == 0:
                    print(f'[{epoch_ind }, {i + 1:5d}] loss: {running_loss / checkpoint_len:.3f}')
                    running_loss = 0.0

    def evaluate(self):
        dataloader = DataLoader(self.test_data, batch_size=64)
        correct = 0
        class_num = self.config.layer_size_list[-1]
        res_mat = np.zeros((class_num, class_num))

        for data in tqdm(dataloader):
            inputs, labels = data

            with torch.no_grad():
                outputs = self.classifier(inputs)
            
            correct += [i for i in range(len(labels)) if outputs[i] == labels[i]]
            for i in range(len(labels)):
                res_mat[outputs[i], labels[i]] += 1

        return correct/len(dataloader), res_mat

class SVMTrainer:
    def __init__(model, training_data, config):
        super.__init__(model, training_data, config)

    def train(self):
        return False
        

def create_trainer(classifier, classifier_config, train_data, test_data):
    if classifier_config.classifier_type == 'neural':
        return NeuralTrainer(classifier, classifier_config, train_data, test_data)
    elif classifier_config.classifier_type == 'svm':
        return SVMTrainer(classifier, classifier_config, train_data, test_data)
    else:
        assert False, 'Classifier ' + classifier_config.classifier_type + ' unknown'
