#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2023
Edited on April 2023 by Melih Kurtaran
"""


###########################################################################
# Import libraries and set up configuration parameters
###########################################################################
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import copy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
##############################################################################
# Functions to be developed by the student
##############################################################################

# Test step
def test_pass(data, target, model, criterion):
    """
    Evaluates a given model on a single batch of data.

    Args:
        data (torch.Tensor): Input data tensor.
        target (torch.Tensor): Target tensor.
        model (torch.nn.Module): PyTorch model to be evaluated.
        criterion (torch.nn.Module): Loss function.

    Returns:
        Tuple containing the loss value and output tensor.
    """

    # set model to evaluation mode
    model.eval()

    # move data and target to device
    data, target = data.to(DEVICE), target.to(DEVICE)

    # perform forward pass
    with torch.no_grad():
        output = model(data)
        loss = criterion(output, target)

    # return loss value and output tensor
    return loss.item(), output


def do_test(model, loaders, criterion):
    """
    Test the performance of the given model on the test set.

    Args:
        model: A PyTorch model.
        loaders: A dictionary with 'test' DataLoader.
        criterion: A PyTorch loss function.

    Returns:
        accuracy: A float representing the accuracy on the test set.
        test_loss: A float representing the average test loss.
        all_outputs: A tensor containing the outputs of the model on the test
        set.
    """

    # Set the model to evaluation mode
    model.eval()

    # create DataLoader for test set
    test_loader = loaders['test']

    # initialize variables
    test_loss = 0.0
    num_correct = 0
    all_outputs = torch.tensor([]).to(DEVICE)

    # evaluate model on test set
    for data, target in test_loader:
        # perform test pass
        loss, output = test_pass(data, target, model, criterion)

        # update loss and accuracy
        test_loss += loss * data.size(0)

        output = output.to(DEVICE)
        target = target.to(DEVICE)

        num_correct += torch.sum(torch.argmax(output, dim=1) == target).item()

        # concatenate all outputs
        all_outputs = torch.cat((all_outputs, output), dim=0)

    # calculate average test loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = num_correct / len(test_loader.dataset)

    # return accuracy, test loss and all outputs
    return accuracy, test_loss, all_outputs


# %%
##############################################################################
# Provided functions
##############################################################################


def inspect_obj(obj, internal=False):
    """Return the attributes (properties and methods) of an object"""

    """Return a dictionary with three elements. The first element has
    'properties' as key and its value is a list of strings with all the
    properties that the dir() function was able to retrieve.
    The second element of the dictionary, with 'methods' key, is the equivalent
    applied to methods.
    The third element is the union of the previous two, and it's key is
    'attributes'.
    You might want to take a look at the 'inspect' library if you need to dig
    deeper. An example of use would be:
    print(inspect_obj(obj)['properties'])

    Parameters
    ----------
    obj :
        TYPE: object
        DESCRIPTION: It can be any object.
    internal :
        TYPE: bool
        DESCRIPTION: If True it also returns the attributes that start with
            underscore.

    Returns
    -------
    output :
        TYPE: Dictionary of two elements of the type list of strings.
        DESCRIPTION. Dictionary with two elements. The first is
            output['properties'] and the second is output['methods']. They list
            the properties and methods respectively.
    """

    dir_obj = []

    # Loop through attributes found by dir(). This first filter is done because
    # sometimes there are attributes that raise an error when called by
    # getattr() due to they haven't been initialized, or due to they have a
    # special behavior.
    for func in dir(obj):
        try:
            _ = getattr(obj, func)
            dir_obj.append(func)
        except BaseException:
            pass

    # Selection of methods and properties
    if internal:
        method_list = [func for func in dir_obj if callable(getattr(obj,
                                                                    func))]
        property_list = [prop for prop in dir_obj if prop not in method_list]
    else:
        method_list = [func for func in dir_obj if callable(
            getattr(obj, func)) and not func.startswith('_')]
        property_list = [prop for prop in dir_obj if
                         prop not in method_list and not prop.startswith('_')]

    return {'properties': property_list, 'methods': method_list,
            'attributes': sorted(property_list + method_list)}


# Train step
def train_pass(data, target, model, optimizer, criterion):
    data, target = data.to(DEVICE), target.to(DEVICE)
    optimizer.zero_grad()
    output = model.forward(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


# Validation step
def valid_pass(data, target, model, criterion):
    data, target = data.to(DEVICE), target.to(DEVICE)
    with torch.no_grad():
        output = model.forward(data)
    loss = criterion(output, target)
    return loss.item()


# Saving the model
def trained_save(filename, model, optimizer, tr_loss_list, vl_loss_list,
                 verbose=True):
    custom_dict = {'model_state_dict': model.state_dict(),
                   'opt_state_dict': optimizer.state_dict(),
                   'tr_loss_list': tr_loss_list,
                   'vl_loss_list': vl_loss_list}
    torch.save(custom_dict, filename)
    if verbose:
        print('Checkpoint saved at epoch {}'.format(len(tr_loss_list)))


# Load the model saved with 'trained_save()'
def trained_load(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    checkpoint.pop('model_state_dict')
    checkpoint.pop('opt_state_dict')

    return model, optimizer, checkpoint


# Training function
def train(n_epochs, loaders, model, optimizer, criterion, filename='model.pt',
          checkpoint={}):

    # Get parameters from checkpoint if available
    if bool(checkpoint):
        tr_loss_list = checkpoint['tr_loss_list']
        vl_loss_list = checkpoint['vl_loss_list']
        valid_loss_min = np.min(vl_loss_list)
        trained_epochs = len(tr_loss_list)
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        tr_loss_list = []
        vl_loss_list = []
        valid_loss_min = np.Inf
        trained_epochs = 0
        best_state_dict = {}

    # Loop through epochs
    for epoch in range(1 + trained_epochs, n_epochs + trained_epochs + 1):
        start_time = time.time()
        model.train()
        train_loss, valid_loss = 0.0, 0.0

        # Training
        for data, target in loaders['train']:
            train_loss += train_pass(data, target, model, optimizer, criterion)
        # Training losses log
        tr_loss_list.append(train_loss / len(loaders['train']))

        # Validation
        model.eval()
        for data, target in loaders['valid']:
            valid_loss += valid_pass(data, target, model, criterion)
        # Validation losses log
        vl_loss_list.append(valid_loss / len(loaders['valid']))

        # Results
        end_time = time.time()
        print('Epoch: {} \tTraining loss: {:.5f} \tValidation loss: {:.5f}\
            \t Time: {:.1f} s'.format(epoch, tr_loss_list[-1],
                                      vl_loss_list[-1], end_time - start_time))

        # Saving best model
        if vl_loss_list[-1] < valid_loss_min:
            best_state_dict = copy.deepcopy(model.state_dict())
            trained_save(filename, model, optimizer,
                         tr_loss_list, vl_loss_list)
            valid_loss_min = vl_loss_list[-1]

    # The best model is returned and the training data are written before
    # exiting
    model.load_state_dict(best_state_dict)
    trained_save(filename, model, optimizer, tr_loss_list, vl_loss_list, False)

    return model, (tr_loss_list, vl_loss_list)


# Traing plot
def plot_checkpoint(checkpoint):
    x = range(1, 1 + len(checkpoint['tr_loss_list']))
    tr_losses = checkpoint['tr_loss_list']
    vl_losses = checkpoint['vl_loss_list']
    tr_max, tr_min = np.max(tr_losses), np.min(tr_losses)
    epoch_min = 1 + np.argmin(vl_losses)
    val_min = np.min(vl_losses)

    plt.plot(x, tr_losses, label='training loss')
    plt.plot(x, vl_losses, label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Losses during training")
    plt.legend()
    plt.annotate('valid min: {:.4f}'.format(val_min),
                 xy=(epoch_min,
                     val_min),
                 xytext=(round(0.75 * len(tr_losses)),
                         3 * (tr_max - tr_min) / 4 + tr_min),
                 arrowprops=dict(facecolor='black',
                                 shrink=0.05),
                 )
    plt.xlim(0, len(tr_losses))
    plt.show()


# %%
##############################################################################
# Test functions
##############################################################################


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Dense layers
        # Three dense layers are defined. The input of the first hidden layer
        # will have as many units as pixels in the image. The output of the
        # last layer will have as many units as classes we want to identify, in
        # this case 10 digits
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=256, bias=True)
        self.fc2 = nn.Linear(in_features=256, out_features=64, bias=True)
        self.fc3 = nn.Linear(in_features=64, out_features=10, bias=True)

        # Activation layers
        self.relu = nn.ReLU()

        # Dropout layers.
        # The parameter p is the probability of each unit to be turned off in
        # the current epoch. We'll see an example shortly. The dropout is now
        # set to disabled.
        self.dropout = nn.Dropout(p=0.0)

    # Definition of forward pass method
    def forward(self, x):
        # The inputs will propagate forward through all the defined layers. The
        # behavior is specified by each function.
        x = x.view(-1, 28 * 28)  # in: b x 28 x 28  out: b x 784
        x = self.relu(self.fc1(x))  # in: b x 784  out: b x 256
        x = self.dropout(x)
        x = self.relu(self.fc2(x))  # in: b x 256  out: b x 64
        x = self.dropout(x)
        x = self.relu(self.fc3(x))  # in: b x 64  out: b x 10
        # Notice how in this case there is no output activation that converts
        # the scores into probabilities.

        return x


# Test test_pass
def test_test_pass():
    # Set seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Define test data
    b = 6
    data = torch.randn(b, 1, 28, 28).to(DEVICE)
    target = torch.randint(0, 10, (b,)).to(DEVICE)

    # Define PyTorch model and criterion
    model = model = Network().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()

    # Call the function to be tested
    loss, output = test_pass(data, target, model, criterion)

    # Check that loss is a scalar float
    assert isinstance(loss, float)

    # Check the loss calculation
    assert np.round(loss, 4) == 2.3448

    # Check that output is a tensor with correct shape
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([b, 10])

    # Check some values
    assert np.round(
        float(torch.sum(output[2, :]) - torch.sum(output[:, 4])), 4) == -1.3886


def test_do_test():
    # Set seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            x = self.fc(x)
            return x

    # Create a toy dataset
    X_test = torch.randn((100, 10))
    y_test = torch.randint(0, 2, (100,))

    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=10)

    # Create an instance of the model
    model = SimpleModel().to(DEVICE)

    # Define a loss function
    criterion = nn.CrossEntropyLoss()

    # Test the function
    accuracy, test_loss, output = do_test(
        model, {'test': test_loader}, criterion)

    # Check that the outputs have the correct shapes
    assert isinstance(accuracy, float)
    assert isinstance(test_loss, float)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (100, 2)

    # Check that the accuracy is in the range [0,1]
    assert 0 <= accuracy <= 1

    # Check that the accuracy is correctly calculated
    assert accuracy == 0.59

    # Check that the test loss is not negative
    assert test_loss >= 0

    # Check that test_loss is correctly calculated
    assert np.round(test_loss, 6) == 0.718312

    # Check some values of output
    assert np.round(
        float(torch.sum(output[2:4, :]) - torch.sum(
            output[4:6, :])), 4) == -0.9756


##############################################################################
# Main block
##############################################################################

if __name__ == "__main__":

    SEED = 4

    test_test_pass()

    test_do_test()

    print("All tests passed!")
