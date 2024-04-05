import torchvision
import torch
from torchvision import transforms

class SplitMnist():
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),     

        ])
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        #flatten the image
        self.X_train = torch.stack([sample[0].view(-1) for sample in mnist_train])
        self.X_test = torch.stack([sample[0].view(-1) for sample in mnist_test])
        self.train_label = torch.tensor([sample[1] for sample in mnist_train])
        self.test_label = torch.tensor([sample[1] for sample in mnist_test])

        self.labels_0 = [0, 2, 4, 6, 8]
        self.labels_1 = [1, 3, 5, 7, 9]
        self.current_task = 0
        self.n_tasks = len(self.labels_0)

    def get_dims(self):
        # Return the number of features and number of classes
        return 784, 2

    def next_task(self):
        if self.current_task >= self.n_tasks:
            raise Exception('All tasks completed already')
        else:
            negative_ids_train = torch.nonzero(self.train_label == self.labels_0[self.current_task]).squeeze(1)
            positive_ids_train = torch.nonzero(self.train_label == self.labels_1[self.current_task]).squeeze(1)
            X_train = torch.cat((self.X_train[negative_ids_train], self.X_train[positive_ids_train]), dim=0)

            y_train = torch.cat((torch.ones(negative_ids_train.shape[0], dtype=torch.float).unsqueeze(1),
                                      torch.zeros(positive_ids_train.shape[0], dtype=torch.float).unsqueeze(1)), dim=0)
            y_train = torch.cat((y_train, 1 - y_train), dim=1)
            #y_train[:,0] contains the labels and y_train[:,1] contains the complementary labels
            negative_ids_test = torch.nonzero(self.test_label == self.labels_0[self.current_task]).squeeze(1)
            positive_ids_test = torch.nonzero(self.test_label == self.labels_1[self.current_task]).squeeze(1)
            X_test = torch.cat((self.X_test[negative_ids_test], self.X_test[positive_ids_test]), dim=0)

            y_test = torch.cat((torch.ones(negative_ids_test.shape[0], dtype=torch.float).unsqueeze(1),
                                     torch.zeros(positive_ids_test.shape[0], dtype=torch.float).unsqueeze(1)), dim=0)
            y_test = torch.cat((y_test, 1 - y_test), dim=1)

            self.current_task += 1

            return X_train, y_train, X_test, y_test
    
    def reset(self):
        self.current_task = 0
    