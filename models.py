import torch
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
torch.manual_seed(0)

# parameter initionalisations
def weight_parameter(shape, init_weights=None):
    if init_weights is not None:
        initial = torch.tensor(init_weights)
    else:
        initial = torch.randn(shape) * 0.1
    return torch.nn.Parameter(initial)

def bias_parameter(shape):
    initial = torch.ones(shape) * 0.1
    return torch.nn.Parameter(initial)

def small_parameter(shape):
    initial = torch.ones(shape) * -6.0
    return torch.nn.Parameter(initial)

class NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, n_train_samples):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_train_samples = n_train_samples
        

    def train(self, X_train, y_train, task_id, n_epochs=1000, batch_size=100, lr=0.001):
        display_epoch = 5
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n_train_samples = X_train.shape[0]
        if batch_size > n_train_samples:
            batch_size = n_train_samples
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(X_batch, y_batch, task_id)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if epoch % display_epoch == 0:
                #print truncated loss
                print(f'Epoch {epoch} Loss: {epoch_loss:.4f}')
        return losses
    def get_weights(self):
        return [param.detach().numpy() for param in self.parameters()]
    
    def predict(self, X_test, task_id):
        return self._predict(X_test, task_id).detach().numpy()
    def predict_prob(self, X_test, task_id):
        return torch.nn.functional.softmax(self._predict(X_test, task_id), dim=1).detach().numpy()

    


class Vanilla_NN(NN):
    def __init__(self, input_dim, hidden_dims, output_dim, n_train_samples, prev_weights=None, learning_rate=0.001):
        super().__init__(input_dim, hidden_dims, output_dim, n_train_samples)
        self.weights = self.create_weights(prev_weights)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def loss_fn(self, x_batch, y_batch, task_id):
        pred = self._predict(x_batch, task_id)
        return -torch.mean(torch.nn.functional.log_softmax(pred, dim=1)[range(len(y_batch)), y_batch])

    def _predict(self, inputs, task_id):
        act = inputs
        for i in range(len(self.hidden_dims)):
            pre = torch.matmul(act, self.weights[i][0]) + self.weights[i][1]
            act = torch.relu(pre)
        pre = torch.matmul(act, self.weights[-1][0][task_id]) + self.weights[-1][1][task_id]
        return pre

    def create_weights(self, prev_weights):
        weights = []
        for i, size in enumerate([self.input_dim] + self.hidden_dims):
            if prev_weights is not None:
                weight = [torch.tensor(w) for w in prev_weights[i]]
            else:
                weight = [weight_parameter((size, self.hidden_dims[i]), init_weights=None),
                          bias_parameter((self.hidden_dims[i],))]
            weights.append(weight)
        weights[-1].append([weight_parameter((self.hidden_dims[-1], self.output_dim), init_weights=None) for _ in range(self.output_dim)])
        weights[-1].append([bias_parameter((self.output_dim,)) for _ in range(self.output_dim)])
        return weights

class MFVI_NN(NN):
    def __init__(self, input_dim, hidden_dims, output_dim, n_train_samples, prev_means=None, prev_log_variances=None, learning_rate=0.001, prior_mean=0, prior_var=1):

        super().__init__(input_dim, hidden_dims, output_dim, n_train_samples)
        self.weights, self.variances = self.create_weights(prev_means, prev_log_variances)
        self.prior_weights, self.prior_variances = self.create_prior(prev_means, prev_log_variances, prior_mean, prior_var)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def loss_fn(self, x_batch, y_batch, task_id):
        kl_term = self._KL_term()
        pred = self._predict(x_batch, task_id)
        likelihood_term = -torch.mean(torch.nn.functional.log_softmax(pred, dim=1)[range(len(y_batch)), y_batch])
        return kl_term + likelihood_term

    def _predict(self, inputs, task_id):
        return self._predict_layer(inputs, task_id)

    def _predict_layer(self, inputs, task_id):
        act = inputs
        for i in range(len(self.hidden_dims)):
            eps_w = torch.randn(self.weights[i][0].shape)
            eps_b = torch.randn(self.weights[i][1].shape)
            weights = eps_w * torch.exp(0.5 * self.variances[i][0]) + self.weights[i][0]
            biases = eps_b * torch.exp(0.5 * self.variances[i][1]) + self.weights[i][1]
            pre = torch.matmul(act, weights) + biases
            act = torch.relu(pre)
        eps_w = torch.randn(self.weights[-1][0][task_id].shape)
        eps_b = torch.randn(self.weights[-1][1][task_id].shape)
        weights = eps_w * torch.exp(0.5 * self.variances[-1][0][task_id]) + self.weights[-1][0][task_id]
        biases = eps_b * torch.exp(0.5 * self.variances[-1][1][task_id]) + self.weights[-1][1][task_id]
        return torch.matmul(act, weights) + biases

    def _KL_term(self):
        kl = 0
        for i in range(len(self.hidden_dims)):
            kl += self._compute_layer_KL_term(self.weights[i], self.variances[i], self.prior_weights[i], self.prior_variances[i])

        no_tasks = len(self.weights[-1][0])
        for i in range(no_tasks):
            kl += self._compute_layer_KL_term(self.weights[-1], self.variances[-1], self.prior_weights[-1][i], self.prior_variances[-1][i])

        return kl

    def _compute_layer_KL_term(self, weights, variances, prior_weights, prior_variances):
        m, v = weights
        m0, v0 = prior_weights
        kl = -0.5 * m.numel()
        kl += 0.5 * torch.sum(torch.log(v0) - v)
        kl += 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
        return kl

    def create_weights(self, prev_means, prev_log_variances):
        weights = []
        variances = []
        for i, size in enumerate([self.input_dim] + self.hidden_dims):
            if prev_means is not None and prev_log_variances is not None:
                weight = [torch.tensor(w) for w in prev_means[i]]
                variance = [torch.tensor(w) for w in prev_log_variances[i]]
            else:
                weight = [weight_parameter((size, self.hidden_dims[i]), init_weights=None),
                          bias_parameter((self.hidden_dims[i],))]
                variance = [small_parameter((size, self.hidden_dims[i])),
                            small_parameter((self.hidden_dims[i],))]
            weights.append(weight)
            variances.append(variance)
        weights[-1][0].append([weight_parameter((self.hidden_dims[-1], self.output_dim), init_weights=None) for _ in range(self.output_dim)])
        weights[-1][1].append([bias_parameter((self.output_dim,)) for _ in range(self.output_dim)])
        variances[-1][0].append([small_parameter((self.hidden_dims[-1], self.output_dim)) for _ in range(self.output_dim)])
        variances[-1][1].append([small_parameter((self.output_dim,)) for _ in range(self.output_dim)])
        return weights, variances

    def create_prior(self, prev_means, prev_log_variances, prior_mean, prior_var):
        prior_weights = []
        prior_variances = []
        for i, size in enumerate([self.input_dim] + self.hidden_dims):
            if prev_means is not None and prev_log_variances is not None:
                weight = [torch.tensor(w) for w in prev_means[i]]
                variance = [torch.tensor(w) for w in prev_log_variances[i]]
            else:
                weight = [torch.tensor(prior_mean) for _ in range(self.hidden_dims[i])],
                bias = [torch.tensor(prior_mean) for _ in range(self.hidden_dims[i])]
                variance = ([torch.tensor(prior_var) for _ in range(self.hidden_dims[i])],[torch.tensor(prior_var) for _ in range(self.hidden_dims[i])])
                prior_weights.append(weight)
            prior_variances.append(variance)
        prior_weights[-1][0].append([torch.tensor(prior_mean) for _ in range(self.output_dim)])
        prior_weights[-1][1].append([torch.tensor(prior_mean) for _ in range(self.output_dim)])
        prior_variances[-1][0].append([torch.tensor(prior_var) for _ in range(self.output_dim)])
        prior_variances[-1][1].append([torch.tensor(prior_var) for _ in range(self.output_dim)])
        return prior_weights, prior_variances
