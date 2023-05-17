from torch.utils.data import DataLoader
import torch.optim as optim

class Trainer:
    def __init__(self, model, training_data, config):
        self.model = model
        self.training_data = training_data
        self.config = config

class NeuralTrainer:
    def __init__(model, training_data, config):
        super.__init__(model, training_data, config)

    def train(self):
        dataloader = DataLoader(self.training_data, batch_size=64, shuffle=True)
        criterion = self.config.criterion_class()
        optimizer = optim.Adam(self.model.parameters())
        checkpoint_len = 100

        for epoch_ind in range(10):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                inputs, labels = data
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % checkpoint_len == 0:
                    print(f'[{epoch_ind }, {i + 1:5d}] loss: {running_loss / checkpoint_len:.3f}')
                    running_loss = 0.0

class SVMTrainer:
    def __init__(model, training_data, config):
        super.__init__(model, training_data, config)

    def train(self):
        

def create_trainer(config, training_data):
    if config.classifier == 'neural':
        return NeuralTrainer(training_data)
    elif config.classifier == 'svm':
        return SVMTrainer(training_data)
    else:
        assert False, 'Classifier ' + config.classifier + ' unknown'
