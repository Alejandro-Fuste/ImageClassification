import torch.nn as nn
import torch.nn.functional as F

'''
Design a CNN architecture which has more than 2 conv layers and more than 1 fully connected layers. It
should make 10 predictions for the 10 classes of CIFAR-10. Train this network on CIFAR-10 for 30 epochs
using cross-entropy loss and SGD optimizer. Report training/testing loss for each epoch in form of plots and
accuracy scores after 30 epochs. Remember you will need a softmax activation after the final fully connected
layer. 
'''


# classification neural network to recognize RGB color images.

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        """
        layer definitions
        """
        # Convolutional layers for task 1 (3 total)
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)

        # Convolutional layers for task 2 (4 total)
        self.conv4 = nn.Conv2d(32, 32, 3)

        # 2 fully connected layers
        self.fc1 = nn.Linear(32 * 3 * 3, 1000)  # task 1
        # self.fc1 = nn.Linear(32 * 1 * 1, 1000)  # task 2
        self.fc2 = nn.Linear(1000, 1000)

        # output layer
        self.fc3 = nn.Linear(1000, 10)  # (task 1 & 2)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

        """
        The following will select the forward pass function based on mode for the ConvNet.
        Based on the task, you have 2 modes available.
        During creation of each ConvNet model, you will assign one of the valid mode.
        This will fix the forward function (and the network graph) for the entire training/testing
        """
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        else:
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    """
    feed-forward functions
    """
    # Task 1 feed-forward function
    def model_1(self, x):
        # ======================================================================
        # 3 convolutional layers + 2 fully connected layer.

        # Convolution + Pooling + Activation
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # Convolution + Pooling + Activation
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        # Convolution + Pooling + Activation
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        # Flatten the input
        x = x.view(-1, self.flatten_features(x))

        # Apply dropout to first fully-connected layer
        x = self.dropout(F.relu(self.fc1(x)))

        # Apply dropout to second fully-connected layer
        x = self.dropout(F.relu(self.fc2(x)))

        # Output layer
        x = self.fc3(x)

        # Apply softmax activation
        x = F.softmax(x, dim=1)

        return x

    # Task 2 feed-forward function
    def model_2(self, x):
        # ======================================================================
        # 4 convolutional layers + 2 fully connected layer.

        # Convolution + Pooling + Activation
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # Convolution + Pooling + Activation
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        # Convolution + Pooling + Activation
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        # Convolution + Pooling + Activation
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))

        # Flatten the input
        x = x.view(-1, self.flatten_features(x))

        # Apply dropout to first fully-connected layer
        x = self.dropout(F.relu(self.fc1(x)))

        # Apply dropout to second fully-connected layer
        x = self.dropout(F.relu(self.fc2(x)))

        # Output layer
        x = self.fc3(x)

        # Apply softmax activation
        x = F.softmax(x, dim=1)

        return x

    """
    flatten features function
    """
    def flatten_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features