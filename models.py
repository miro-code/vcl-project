import torch
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
torch.manual_seed(0)

#TODO: Implement shared head

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

def KL_of_gaussians(q_mean, q_logvar, p_mean, p_logvar):
    return 0.5 * (p_logvar - q_logvar + (torch.exp(q_logvar) + (q_mean - p_mean)**2) / torch.exp(p_logvar) - 1)

class NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, n_train_samples):
        super().__init__()
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
                loss = self.calculate_loss(X_batch, y_batch, task_id)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if epoch % display_epoch == 0:
                #print truncated loss
                print(f'Epoch {epoch} Loss: {epoch_loss:.4f}')
        return losses
    
    def get_parameters(self):
        return self.params
        
    def predict(self, inputs, task_id):
        raise NotImplementedError

class MLP(NN):
    def __init__(self, input_dim, hidden_dims, output_dim, n_train_samples, lr=0.001):
        super().__init__(input_dim, hidden_dims, output_dim, n_train_samples)
        self.weights, self.biases, self.weights_last, self.biases_last = self.init_weights()
        self.params = [[self.weights, self.biases, self.weights_last, self.biases_last], None]

    def predict(self, inputs, task_id):
        activations = inputs
        for i in range(len(self.hidden_dims)):
            weights = self.weights[i]
            biases = self.biases[i]
            raw_activations = torch.matmul(activations, weights) + biases
            activations = torch.relu(raw_activations)
        logits = torch.matmul(activations, self.weights_last[task_id]) + self.biases_last[task_id]
        return logits

    def predict_proba(self, X_test, task_id):
        return torch.nn.functional.softmax(self.predict(X_test, task_id), dim=1)
    
    def loss_fn(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels)

    def init_weights(self):
        self.layer_dims = deepcopy(self.hidden_dims)
        self.layer_dims.append(self.output_dim)
        self.layer_dims.insert(0, self.input_dim)
        n_layers = len(self.layer_dims) - 1
        
        weights = []
        biases = []
        weights_last = [] #note that this only ever has one element we extend it for BNNs later
        biases_last = []

        #Iterate over layers except the last
        for i in range(n_layers - 1):
            weights.append([weight_parameter((self.layer_dims[i], self.layer_dims[i+1]))])
            biases.append([bias_parameter((self.layer_dims[i+1],))])
        #Last layer
        weights_last.append(weight_parameter((self.layer_dims[-2], self.layer_dims[-1])))
        biases_last.append(bias_parameter((self.layer_dims[-1],)))
        
        return weights, biases, weights_last, biases_last

class BNN(NN):
    def __init__(self, input_dim, hidden_dims, output_dim, n_train_samples, n_pred_samples = 100, previous_means=None, previous_log_variances=None, lr=0.001, prior_mean=0, prior_var=1):
        #previous means is supplied as [weight_means, bias_means, weight_last_means, bias_last_means]
        #previous log variances is supplied equivalently
        #Note that variances are provided as log(variances)
        super().__init__(input_dim, hidden_dims, output_dim, n_train_samples)
        means, variances = self.init_weights(previous_means, previous_log_variances)
        self.weight_means, self.bias_means, self.weight_last_means, self.bias_last_means = means
        self.weight_variances, self.bias_variances, self.weight_last_variances, self.bias_last_variances = variances
        self.params = [means, variances]

        means, variances = self.create_prior(previous_means, previous_log_variances, prior_mean, prior_var) 
        self.prior_weight_means, self.prior_bias_means, self.prior_weight_last_means, self.prior_bias_last_means = means
        self.prior_weight_variances, self.prior_bias_variances, self.prior_weight_last_variances, self.prior_bias_last_variances = variances

        self.n_layers = len(self.layer_dims) - 1
        self.n_train_samples = n_train_samples
        self.n_pred_samples = n_pred_samples

        

    def calculate_loss(self, x_batch, y_batch, task_id):
        kl_term = self.KL_term()
        pred = self.predict(x_batch, task_id)
        likelihood_term = -torch.mean(torch.nn.functional.log_softmax(pred, dim=1)[range(len(y_batch)), y_batch])
        return kl_term + likelihood_term
    
    def predict(self, inputs, task_id):
        return self._predict(inputs, task_id, self.n_pred_samples)
    
    def predict_proba(self, X_test, task_id):
        #TODO: Check if the dim is correct
        return torch.nn.functional.softmax(self.predict(X_test, task_id), dim=1)
    

    def _predict(self, inputs, task_id, n_samples):
        expanded_inputs = inputs.unsqueeze(0) #size: 1 x batch_size x input_dim = 1 x 64 x 784
        activations = expanded_inputs.repeat(n_samples, 1, 1) #size: n_pred_samples x batch_size x input_dim = 100 x 64 x 784
        for i in range(self.n_layers - 1):
            input_dim = self.layer_dims(i)
            output_dim = self.layer_dims(i+1)
            weight_epsilon = torch.randn(n_samples, input_dim, output_dim) #size: n_pred_samples x input_dim x output_dim
            bias_epsilon = torch.randn(n_samples, 1, output_dim) #size: n_pred_samples x 1 x output_dim
            #we use * 0.5 for the reparameterisation trick: taking the square root of the variance is the std
            weights = weight_epsilon * torch.exp(0.5 * self.weight_variances[i]) + self.weight_means[i] #size: TODO:(?) n_pred_samples x batch_size x output_dim 
            biases = bias_epsilon * torch.exp(0.5 * self.bias_variances[i]) + self.bias_means[i]
            raw_activations = torch.matmul(activations, weights) + biases 
            activations = torch.relu(raw_activations) 
        input_dim = self.layer_dims(-2)
        output_dim = self.layer_dims(-1)
        weight_epsilon = torch.randn(n_samples, input_dim, output_dim)
        bias_epsilon = torch.randn(n_samples, 1, output_dim)

        weights = weight_epsilon * torch.exp(0.5 * self.weight_last_variances[task_id]) + self.weight_last_means[task_id]
        biases = bias_epsilon * torch.exp(0.5 * self.bias_last_variances[task_id]) + self.bias_last_means[task_id]
        #TODO: from the original code; check if this is correct 
        activations = activations.unsqueeze(3)
        weights = weights.unsqueeze(1)
        logits = torch.sum(activations * weights, dim=2) + biases
        return logits

    def log_likelihood_loss(self, inputs, targets, task_id):
        prediction = self._predict(inputs, task_id, self.n_train_samples) #TODO: Why is this n_train_samples?
        targets = targets.unsqueeze(0).repeat(self.n_train_samples, 1, 1)
        log_likelihood = torch.nn.functional.cross_entropy(prediction, targets)
        return log_likelihood

    def KL_loss(self):
        loss = 0
        for i in range(self.n_layers - 1):
            loss += torch.sum(KL_of_gaussians(self.weight_means[i], self.weight_variances[i], self.prior_weight_means[i], self.prior_weight_variances[i]))
            loss += torch.sum(KL_of_gaussians(self.bias_means[i], self.bias_variances[i], self.prior_bias_means[i], self.prior_bias_variances[i]))
        loss += torch.sum(KL_of_gaussians(self.weight_last_means[0], self.weight_last_variances[0], self.prior_weight_last_means[0], self.prior_weight_last_variances[0]))
        loss += torch.sum(KL_of_gaussians(self.bias_last_means[0], self.bias_last_variances[0], self.prior_bias_last_means[0], self.prior_bias_last_variances[0]))
        return loss

    def init_weights(self, previous_means, previous_log_variances):
        #previous means is supplied as [weight_means, bias_means, weight_last_means, bias_last_means]
        #previous log variances is supplied equivalently
        
        #Note that the first task is trained to ML so there will be no variance but means.

        weight_means = []
        bias_means = []
        weight_last_means = []
        bias_last_means = []

        weight_variances = []
        bias_variances = []
        weight_last_variances = []
        bias_last_variances = []

        self.layer_dims = deepcopy(self.hidden_dims)
        self.layer_dims.append(self.output_dim)
        self.layer_dims.insert(0, self.input_dim)
        n_layers = len(self.layer_dims) - 1
        
        for i in range(n_layers - 1):
            if previous_means is None:
                weight_mean = weight_parameter((self.layer_dims[i], self.layer_dims[i+1]))
                bias_mean = bias_parameter((self.layer_dims[i+1],))
                weight_variance = small_parameter((self.layer_dims[i], self.layer_dims[i+1])) 
                bias_variance = small_parameter((self.layer_dims[i+1],))
            else:
                weight_mean = torch.nn.Parameter(previous_means[0][i])
                bias_mean = torch.nn.Parameter(previous_means[1][i])
                if(previous_log_variances is None):
                    weight_variance = small_parameter((self.layer_dims[i], self.layer_dims[i+1]))
                    bias_variance = small_parameter((self.layer_dims[i+1],))
                else:
                    weight_variance = torch.nn.Parameter(previous_log_variances[0][i])
                    bias_variance = torch.nn.Parameter(previous_log_variances[1][i])
            weight_means.append(weight_mean)
            bias_means.append(bias_mean)
            weight_variances.append(weight_variance)
            bias_variances.append(bias_variance)
        
        if(previous_log_variances is not None and previous_means is not None):
            previous_weight_last_means = torch.tensor(previous_means[2])
            previous_bias_last_means = torch.tensor(previous_means[3])
            previous_weight_last_variances = torch.tensor(previous_log_variances[2])
            previous_bias_last_variances = torch.tensor(previous_log_variances[3])
            n_previous_tasks = len(previous_weight_last_means)
            for i in range(n_previous_tasks):
                weight_last_means.append(torch.nn.Parameter(previous_weight_last_means[i]))
                bias_last_means.append(torch.nn.Parameter(previous_bias_last_means[i]))
                weight_last_variances.append(torch.nn.Parameter(previous_weight_last_variances[i]))
                bias_last_variances.append(torch.nn.Parameter(previous_bias_last_variances[i]))
        
        if(previous_log_variances is None and previous_means is not None):
            weight_last_means.append(torch.nn.Parameter(previous_means[2][0]))
            bias_last_means.append(torch.nn.Parameter(previous_means[3][0]))
        else:
            weight_last_means.append(weight_parameter((self.layer_dims[-2], self.layer_dims[-1])))
            bias_last_means.append(bias_parameter((self.layer_dims[-1],)))
        weight_last_variances.append(small_parameter((self.layer_dims[-2], self.layer_dims[-1])))
        bias_last_variances.append(small_parameter((self.layer_dims[-1],)))
        return [weight_means, bias_means, weight_last_means, bias_last_means], [weight_variances, bias_variances, weight_last_variances, bias_last_variances]

    def create_prior(self, previous_means, previous_variances, prior_mean, prior_var):
        #previous means is supplied as [weight_means, bias_means, weight_last_means, bias_last_means]
        #previous log variances is supplied equivalently
        self.layer_dims = deepcopy(self.hidden_dims)
        self.layer_dims.append(self.output_dim)
        self.layer_dims.insert(0, self.input_dim)
        n_layers = len(self.layer_dims) - 1
        weight_means = []
        bias_means = []
        weight_last_means = []
        bias_last_means = []
        weight_variances = []
        bias_variances = []
        weight_last_variances = []
        bias_last_variances = []
        
        for i in range(n_layers - 1):
            if(previous_variances is not None and previous_means is not None):
                weight_mean = torch.nn.Parameter(previous_means[0][i])
                bias_mean = torch.nn.Parameter(previous_means[1][i])
                weight_variance = torch.nn.Parameter(previous_variances[0][i])
                bias_variance = torch.nn.Parameter(previous_variances[1][i])
            else:
                weight_mean = torch.nn.Parameter(prior_mean)
                bias_mean = torch.nn.Parameter(prior_mean)
                weight_variance = torch.nn.Parameter(prior_var)
                bias_variance = torch.nn.Parameter(prior_var)
            weight_means.append(weight_mean)
            bias_means.append(bias_mean)
            weight_variances.append(weight_variance)
            bias_variances.append(bias_variance)

        if(previous_variances is not None and previous_means is not None):
            previous_weight_last_means = torch.tensor(previous_means[2])
            previous_bias_last_means = torch.tensor(previous_means[3])
            previous_weight_last_variances = torch.tensor(previous_variances[2])
            previous_bias_last_variances = torch.tensor(previous_variances[3])
            n_previous_tasks = len(previous_weight_last_means)
            for i in range(n_previous_tasks):
                weight_last_means.append(torch.tensor(previous_weight_last_means[i]))
                bias_last_means.append(torch.tensor(previous_bias_last_means[i]))
                weight_last_variances.append(torch.tensor(previous_weight_last_variances[i]))
                bias_last_variances.append(torch.tensor(previous_bias_last_variances[i]))
        else:
            weight_last_means.append(prior_mean)
            bias_last_means.append(prior_mean)
            weight_last_variances.append(prior_var)
            bias_last_variances.append(prior_var)

        return [weight_means, bias_means, weight_last_means, bias_last_means], [weight_variances, bias_variances, weight_last_variances, bias_last_variances]