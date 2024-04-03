import torchvision
import torch

class SplitMnist():
    def __init__(self):
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)

        self.X_train = torch.cat([img.unsqueeze(0) for img, _ in mnist_train], dim=0).view(-1, 784).float()
        self.X_test = torch.cat([img.unsqueeze(0) for img, _ in mnist_test], dim=0).view(-1, 784).float()
        self.train_label = torch.tensor([label for _, label in mnist_train]).long()
        self.test_label = torch.tensor([label for _, label in mnist_test]).long()

        self.labels_0 = [0, 2, 4, 6, 8]
        self.labels_1 = [1, 3, 5, 7, 9]
        self.current_task = 0
        self.n_tasks = len(self.labels_0)

    def get_dims(self):
        # Return the number of features and number of classes
        return 784, 2

    def get_task(self):
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
        