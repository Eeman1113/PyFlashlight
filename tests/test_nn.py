import unittest
import pyflashlight
from pyflashlight.utils import utils_unittests as utils
import torch
import os

class TestNNModuleLoss(unittest.TestCase):

    def setUp(self):
        self.device = os.environ.get('device')
        if self.device is None or self.device != 'cuda':
            self.device = 'cpu'

        print(f"Running tests on: {self.device}")

    def test_mse_loss(self):
        """
        Test the MSELoss
        """
        loss_fn_pyflashlight = pyflashlight.nn.MSELoss()
        loss_fn_torch = torch.nn.MSELoss()

        # Test case 1: Predictions and labels are equal
        predictions_pyflashlight = pyflashlight.Tensor([[1.1, 2, 3, 4], [1.1, 2, 3, 4]]).to(self.device)
        labels_pyflashlight = pyflashlight.Tensor([[1.1, 2, 3, 4], [1.1, 2, 3, 3]]).to(self.device)
        loss_pyflashlight = loss_fn_pyflashlight.forward(predictions_pyflashlight, labels_pyflashlight)
        loss_torch_result = utils.to_torch(loss_pyflashlight).to(self.device)

        predictions_torch = torch.tensor([[1.1, 2, 3, 4], [1.1, 2, 3, 4]]).to(self.device)
        labels_torch = torch.tensor([[1.1, 2, 3, 4], [1.1, 2, 3, 3]]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)        
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))
        
        # Test case 2: Predictions and labels are different
        predictions_pyflashlight = pyflashlight.Tensor([1.1, 2, 3, 4]).to(self.device)
        labels_pyflashlight = pyflashlight.Tensor([4, 3, 2.1, 1]).to(self.device)
        loss_pyflashlight = loss_fn_pyflashlight.forward(predictions_pyflashlight, labels_pyflashlight)
        loss_torch_result = utils.to_torch(loss_pyflashlight).to(self.device)

        predictions_torch = torch.tensor([1.1, 2, 3, 4]).to(self.device)
        labels_torch = torch.tensor([4, 3, 2.1, 1]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)        
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))

    def test_cross_entropy_loss(self):
        """
        Test the CrossEntropyLoss
        """
        loss_fn_pyflashlight = pyflashlight.nn.CrossEntropyLoss()
        loss_fn_torch = torch.nn.CrossEntropyLoss()

        # Test case 1: Single class, single sample
        predictions_pyflashlight = pyflashlight.Tensor([2.0, 1.0, 0.1]).to(self.device)
        labels_pyflashlight = pyflashlight.Tensor([0]).to(self.device)
        loss_pyflashlight = loss_fn_pyflashlight.forward(predictions_pyflashlight, labels_pyflashlight)

        loss_torch_result = utils.to_torch(loss_pyflashlight).to(self.device)

        predictions_torch = torch.tensor([2.0, 1.0, 0.1]).to(self.device)
        labels_torch = torch.tensor(0).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)

        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))

        # Test case 2: Multiple classes, multiple samples
        predictions_pyflashlight = pyflashlight.Tensor([[0.5, 1.5, 2.5], [1.0, 2.0, 3.0]]).to(self.device)
        labels_pyflashlight = pyflashlight.Tensor([2, 1]).to(self.device)
        loss_pyflashlight = loss_fn_pyflashlight.forward(predictions_pyflashlight, labels_pyflashlight)
        loss_torch_result = utils.to_torch(loss_pyflashlight).to(self.device)
        

        predictions_torch = torch.tensor([[0.5, 1.5, 2.5], [1.0, 2.0, 3.0]]).to(self.device)
        labels_torch = torch.tensor([2, 1]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))
        
        # Test case 3: Edge case - all predictions are zero
        predictions_pyflashlight = pyflashlight.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).to(self.device)
        labels_pyflashlight = pyflashlight.Tensor([1, 2]).to(self.device)
        loss_pyflashlight = loss_fn_pyflashlight.forward(predictions_pyflashlight, labels_pyflashlight)
        loss_torch_result = utils.to_torch(loss_pyflashlight).to(self.device)
        
        predictions_torch = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).to(self.device)
        labels_torch = torch.tensor([1, 2]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))

        # Test case 4: Class probabilities instead of class index
        predictions_pyflashlight = pyflashlight.Tensor([0.5, 0.2, 0.1]).to(self.device)
        labels_pyflashlight = pyflashlight.Tensor([1., 0, 0]).to(self.device)
        loss_pyflashlight = loss_fn_pyflashlight.forward(predictions_pyflashlight, labels_pyflashlight)
        loss_torch_result = utils.to_torch(loss_pyflashlight).to(self.device)
        
        predictions_torch = torch.tensor([0.5, 0.2, 0.1]).to(self.device)
        labels_torch = torch.tensor([1., 0, 0]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))

        # Test case 4: Batched class probabilities instead of class index
        predictions_pyflashlight = pyflashlight.Tensor([[0.5, 0.2, 0.1], [0.1, 0.5, 0.7]]).to(self.device)
        labels_pyflashlight = pyflashlight.Tensor([[1., 0, 0], [0, 1, 0]]).to(self.device)
        loss_pyflashlight = loss_fn_pyflashlight.forward(predictions_pyflashlight, labels_pyflashlight)
        loss_torch_result = utils.to_torch(loss_pyflashlight).to(self.device)

        predictions_torch = torch.tensor([[0.5, 0.2, 0.1], [0.1, 0.5, 0.7]]).to(self.device)
        labels_torch = torch.tensor([[1., 0, 0], [0, 1, 0]]).to(self.device)
        loss_torch_expected = loss_fn_torch(predictions_torch, labels_torch)
        
        self.assertTrue(utils.compare_torch(loss_torch_result, loss_torch_expected))


class TestNNModuleActivationFn(unittest.TestCase):

    def setUp(self):
        self.device = os.environ.get('device')
        if self.device is None or self.device != 'cuda':
            self.device = 'cpu'

    def test_sigmoid_activation(self):
        """
        Test Sigmoid activation function
        """
        sigmoid_fn_pyflashlight = pyflashlight.nn.Sigmoid()
        sigmoid_fn_torch = torch.nn.Sigmoid()

        # Test case 1: Positive input
        x = pyflashlight.Tensor([[1, 2, 3]]).to(self.device)
        sigmoid_pyflashlight = sigmoid_fn_pyflashlight.forward(x)
        sigmoid_torch_result = utils.to_torch(sigmoid_pyflashlight).to(self.device)

        x = torch.tensor([[1, 2, 3]]).to(self.device)
        sigmoid_torch_expected = sigmoid_fn_torch.forward(x)

        self.assertTrue(utils.compare_torch(sigmoid_torch_result, sigmoid_torch_expected))

        # Test case 1: Negative input
        x = pyflashlight.Tensor([-1, 2, -3]).to(self.device)
        sigmoid_pyflashlight = sigmoid_fn_pyflashlight.forward(x)
        sigmoid_torch_result = utils.to_torch(sigmoid_pyflashlight).to(self.device)

        x = torch.tensor([-1, 2, -3]).to(self.device)
        sigmoid_torch_expected = sigmoid_fn_torch.forward(x)

        self.assertTrue(utils.compare_torch(sigmoid_torch_result, sigmoid_torch_expected))

        # Test case 1: Zero input
        x = pyflashlight.Tensor([0, 0, 0]).to(self.device)
        sigmoid_pyflashlight = sigmoid_fn_pyflashlight.forward(x)
        sigmoid_torch_result = utils.to_torch(sigmoid_pyflashlight).to(self.device)

        x = torch.tensor([0, 0, 0]).to(self.device)
        sigmoid_torch_expected = sigmoid_fn_torch.forward(x)

        self.assertTrue(utils.compare_torch(sigmoid_torch_result, sigmoid_torch_expected))

    def test_softmax_activation(self):
        """
        Test Softmax activation function
        """    

        # Test different axes
        axes = [0, 1, 2, -1]

        # Define the input tensors for different test cases
        test_cases = [
            (pyflashlight.Tensor([[[1., 2, 3], [4, 5, 6]]]), torch.tensor([[[1., 2, 3], [4, 5, 6]]])),
            (pyflashlight.Tensor([[[1., -1, 0], [2, -2, 0]]]), torch.tensor([[[1., -1, 0], [2, -2, 0]]])),
            (pyflashlight.Tensor([[[0., 0, 0], [0, 0, 0]]]), torch.tensor([[[0., 0, 0], [0, 0, 0]]]))
        ]

        for dim in axes:
            softmax_fn_pyflashlight = pyflashlight.nn.Softmax(dim=dim)
            softmax_fn_torch = torch.nn.Softmax(dim=dim)

            for pyflashlight_input, torch_input in test_cases:
                # Move tensors to the correct device
                pyflashlight_input = pyflashlight_input.to(self.device)
                torch_input = torch_input.to(self.device)

                # Forward pass using pyflashlight
                softmax_pyflashlight = softmax_fn_pyflashlight.forward(pyflashlight_input)
                softmax_torch_result = utils.to_torch(softmax_pyflashlight).to(self.device)

                # Forward pass using torch
                softmax_torch_expected = softmax_fn_torch.forward(torch_input)

                # Compare the results
                self.assertTrue(utils.compare_torch(softmax_torch_result, softmax_torch_expected))

