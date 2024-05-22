import unittest
import pyflashlight
from pyflashlight.utils import utils_unittests as utils
import torch
import sys
import os

class TestTensorOperations(unittest.TestCase):

    def setUp(self):
        self.device = os.environ.get('device')
        if self.device is None or self.device != 'cuda':
            self.device = 'cpu'

        print(f"Running tests on: {self.device}")

    def test_creation_and_conversion(self):
        """
        Test creation and convertion of pyflashlight tensor to pytorch
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor = utils.to_torch(pyflashlight_tensor)
        self.assertTrue(torch.is_tensor(torch_tensor))

    def test_addition(self):
        """
        Test addition two tensors: tensor1 + tensor2
        """
        pyflashlight_tensor1 = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor1 + pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_addition_broadcasted(self):
        """
        Test addition of two tensors with broadcasting: tensor1 + tensor2
        """
        pyflashlight_tensor1 = pyflashlight.Tensor([[[1, 2, 3], [4, 5, 6]]]).to(self.device)  # Shape (1, 2, 3)
        pyflashlight_tensor2 = pyflashlight.Tensor([1, 1, 1]).to(self.device)  # Shape (3)
        pyflashlight_result = pyflashlight_tensor1 + pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]]).to(self.device)  # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([1, 1, 1]).to(self.device)  # Shape (3)
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        pyflashlight_tensor1 = pyflashlight.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)  # Shape (1, 2, 3)
        pyflashlight_tensor2 = pyflashlight.Tensor([[10, 10], [5, 6]]).to(self.device)  # Shape (3)
        pyflashlight_result = pyflashlight_tensor1 + pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)
        
        torch_tensor1 = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)  # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([[[10, 10], [5, 6]]]).to(self.device)  # Shape (3)
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # reversed order broadcasting
        pyflashlight_tensor1 = pyflashlight.Tensor([[0, 2]]).to(self.device) 
        pyflashlight_tensor2 = pyflashlight.Tensor([[3, 4], [5, -1]]).to(self.device) 
        pyflashlight_result = pyflashlight_tensor1 + pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[0, 2]]).to(self.device)  
        torch_tensor2 = torch.tensor([[3, 4], [5, -1]]).to(self.device) 
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        pyflashlight_result = pyflashlight_tensor2 + pyflashlight_tensor1
        torch_expected = torch_tensor2 + torch_tensor1

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))



    def test_subtraction(self):
        """
        Test subtraction of two tensors: tensor1 - tensor2
        """
        pyflashlight_tensor1 = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor1 - pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        torch_expected = torch_tensor1 - torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_broadcasting_subtraction(self):
        """
        Test subtraction of two tensors with broadcasting: tensor1 - tensor2
        """
        pyflashlight_tensor1 = pyflashlight.Tensor([[[1, 2, 3], [4, 5, 6]]]).to(self.device)  # Shape (1, 2, 3)
        pyflashlight_tensor2 = pyflashlight.Tensor([1, 1, 1]).to(self.device)  # Shape (3)
        pyflashlight_result = pyflashlight_tensor1 - pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]]).to(self.device)  # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([1, 1, 1]).to(self.device)  # Shape (3)
        torch_expected = torch_tensor1 - torch_tensor2 

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # reversed order broadcasting
        pyflashlight_result = pyflashlight_tensor2 - pyflashlight_tensor1
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_expected = torch_tensor2 - torch_tensor1

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_division_by_scalar(self):
        """
        Test division of a tensor by a scalar: tensor / scalar
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]).to(self.device)
        scalar = 2
        pyflashlight_result = pyflashlight_tensor / scalar
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]).to(self.device)
        torch_expected = torch_tensor / scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_scalar_division_by_tensor(self):
        """
        Test scalar division by a tensor: scalar / tensor
        """
        scalar = 10
        pyflashlight_tensor = pyflashlight.Tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]).to(self.device)
        pyflashlight_result = scalar / pyflashlight_tensor
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]).to(self.device)
        torch_expected = scalar / torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_matrix_multiplication(self):
        """
        Test matrix multiplication: tensor1 @ tensor2
        """
        pyflashlight_tensor1 = pyflashlight.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor1 @ pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]]).to(self.device)
        torch_expected = torch_tensor1 @ torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_elementwise_multiplication_by_scalar(self):
        """
        Test elementwise multiplication of a tensor by a scalar: tensor * scalar
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        scalar = 2
        pyflashlight_result = pyflashlight_tensor * scalar
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch_tensor * scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_elementwise_multiplication_by_tensor(self):
        """
        Test elementwise multiplication of two tensors: tensor1 * tensor2
        """
        pyflashlight_tensor1 = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[[2, 2], [2, 2]], [[2, 2], [2, 2]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor1 * pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[2, 2], [2, 2]], [[2, 2], [2, 2]]]).to(self.device)
        torch_expected = torch_tensor1 * torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_reshape(self):
        """
        Test reshaping of a tensor: tensor.reshape(shape)
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        new_shape = [2, 4]
        pyflashlight_result = pyflashlight_tensor.reshape(new_shape)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch_tensor.reshape(new_shape)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_unsqueeze(self):
        """
        Test unsqueeze operation on a tensor
        """
        pyflashlight_tensor = pyflashlight.Tensor([[1, 2], [3, 4]]).to(self.device)
        
        # Unsqueeze at dim=0
        pyflashlight_unsqueeze_0 = pyflashlight_tensor.unsqueeze(0)
        torch_unsqueeze_0 = utils.to_torch(pyflashlight_unsqueeze_0).to(self.device)
        torch_tensor = torch.tensor([[1, 2], [3, 4]]).to(self.device)
        torch_expected_0 = torch_tensor.unsqueeze(0)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_0, torch_expected_0))

        # Unsqueeze at dim=1
        pyflashlight_unsqueeze_1 = pyflashlight_tensor.unsqueeze(1)
        torch_unsqueeze_1 = utils.to_torch(pyflashlight_unsqueeze_1).to(self.device)
        torch_expected_1 = torch_tensor.unsqueeze(1)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_1, torch_expected_1))

        # Unsqueeze at dim=2
        pyflashlight_unsqueeze_2 = pyflashlight_tensor.unsqueeze(2)
        torch_unsqueeze_2 = utils.to_torch(pyflashlight_unsqueeze_2).to(self.device)
        torch_expected_2 = torch_tensor.unsqueeze(2)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_2, torch_expected_2))

        # Unsqueeze at dim=-1
        pyflashlight_unsqueeze_neg_1 = pyflashlight_tensor.unsqueeze(-1)
        torch_unsqueeze_neg_1 = utils.to_torch(pyflashlight_unsqueeze_neg_1).to(self.device)
        torch_expected_neg_1 = torch_tensor.unsqueeze(-1)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_neg_1, torch_expected_neg_1))

        # Unsqueeze at dim=-2
        pyflashlight_unsqueeze_neg_2 = pyflashlight_tensor.unsqueeze(-2)
        torch_unsqueeze_neg_2 = utils.to_torch(pyflashlight_unsqueeze_neg_2).to(self.device)
        torch_expected_neg_2 = torch_tensor.unsqueeze(-2)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_neg_2, torch_expected_neg_2))

    def test_squeeze(self):
        """
        Test squeeze operation on a tensor
        """
        # Create a tensor with some dimensions of size 1
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, 4]]]).to(self.device)  # shape [1, 2, 2]
        
        # Squeeze at dim=0
        pyflashlight_squeeze_0 = pyflashlight_tensor.squeeze(0)
        torch_squeeze_0 = utils.to_torch(pyflashlight_squeeze_0).to(self.device)
        torch_tensor = torch.tensor([[[1, 2], [3, 4]]]).to(self.device)
        torch_expected_0 = torch_tensor.squeeze(0)
        self.assertTrue(utils.compare_torch(torch_squeeze_0, torch_expected_0))

        # Create a tensor with a dimension of size 1 in the middle
        pyflashlight_tensor_middle_1 = pyflashlight.Tensor([[[1, 2]], [[3, 4]]]).to(self.device)  # shape [2, 1, 2]
        
        # Squeeze at dim=1
        pyflashlight_squeeze_1 = pyflashlight_tensor_middle_1.squeeze(1)
        torch_squeeze_1 = utils.to_torch(pyflashlight_squeeze_1).to(self.device)
        torch_tensor_middle_1 = torch.tensor([[[1, 2]], [[3, 4]]]).to(self.device)
        torch_expected_1 = torch_tensor_middle_1.squeeze(1)
        self.assertTrue(utils.compare_torch(torch_squeeze_1, torch_expected_1))

        # Squeeze at dim=-2 (same as dim=1 in this case)
        pyflashlight_squeeze_neg_2 = pyflashlight_tensor_middle_1.squeeze(-2)
        torch_squeeze_neg_2 = utils.to_torch(pyflashlight_squeeze_neg_2).to(self.device)
        torch_expected_neg_2 = torch_tensor_middle_1.squeeze(-2)
        self.assertTrue(utils.compare_torch(torch_squeeze_neg_2, torch_expected_neg_2))

        # Squeeze all dimensions of size 1 (None)
        pyflashlight_tensor_all_1 = pyflashlight.Tensor([[[[1, 2], [3, 4]]]]).to(self.device)  # shape [1, 1, 2, 2]
        pyflashlight_squeeze_all = pyflashlight_tensor_all_1.squeeze()
        torch_squeeze_all = utils.to_torch(pyflashlight_squeeze_all).to(self.device)
        torch_tensor_all_1 = torch.tensor([[[[1, 2], [3, 4]]]]).to(self.device)
        torch_expected_all = torch_tensor_all_1.squeeze()
        self.assertTrue(utils.compare_torch(torch_squeeze_all, torch_expected_all))

        # Squeeze no dimensions (no dimensions of size 1)
        pyflashlight_tensor_no_1 = pyflashlight.Tensor([[1, 2], [3, 4]]).to(self.device)  # shape [2, 2]
        pyflashlight_squeeze_none = pyflashlight_tensor_no_1.squeeze()
        torch_squeeze_none = utils.to_torch(pyflashlight_squeeze_none).to(self.device)
        torch_tensor_no_1 = torch.tensor([[1, 2], [3, 4]]).to(self.device)
        torch_expected_none = torch_tensor_no_1.squeeze()
        self.assertTrue(utils.compare_torch(torch_squeeze_none, torch_expected_none))


    def test_transpose(self):
        """
        Test transposition of a tensor: tensor.transpose(dim1, dim2)
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        dim1, dim2 = 0, 2
        pyflashlight_result = pyflashlight_tensor.transpose(dim1, dim2)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch_tensor.transpose(dim1, dim2)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_logarithm(self):
        """
        Test elementwise logarithm of a tensor: tensor.log()
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.log()
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.log(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_sum(self):
        """
        Test summation of a tensor: tensor.sum()
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.sum()
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.sum(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_sum_axis(self):
        """
        Test summation of a tensor along a specific axis without keeping the dimensions
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.sum(axis=1)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.sum(torch_tensor, dim=1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # negative axis

        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.sum(axis=-2)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.sum(torch_tensor, dim=-2)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_sum_axis_keepdim(self):
        """
        Test summation of a tensor along a specific axis with keepdim=True
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.sum(axis=1, keepdim=True)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.sum(torch_tensor, dim=1, keepdim=True)
        
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_max(self):
        """
        Test max of a tensor: tensor.max()
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.max()
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.max(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_max_axis(self):
        """
        Test max of a tensor along a specific axis without keeping the dimensions
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.max(axis=1)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.max(torch_tensor, dim=1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # negative axis

        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.max(axis=-1)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.max(torch_tensor, dim=-1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_max_axis_keepdim(self):
        """
        Test max of a tensor along a specific axis with keepdim=True
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.max(axis=1, keepdim=True)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.max(torch_tensor, dim=1, keepdim=True)
        
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_min(self):
        """
        Test min of a tensor: tensor.min()
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.min()
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.min(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_min_axis(self):
        """
        Test min of a tensor along a specific axis without keeping the dimensions
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.min(axis=1)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.min(torch_tensor, dim=1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # negative axis

        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.min(axis=-1)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.min(torch_tensor, dim=-1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_min_axis_keepdim(self):
        """
        Test min of a tensor along a specific axis with keepdim=True
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.min(axis=1, keepdim=True)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected, _ = torch.min(torch_tensor, dim=1, keepdim=True)
        
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_transpose_T(self):
        """
        Test transposition of a tensor: tensor.T
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.T
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch.transpose(torch_tensor, 0, 2)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_matmul(self):
        """
        Test matrix multiplication: MxP = NxM @ MxP
        """
        # Creating batched tensors for pyflashlight
        pyflashlight_tensor1 = pyflashlight.Tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]]).to(self.device)

        pyflashlight_result = pyflashlight_tensor1 @ pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        # Converting to PyTorch tensors for comparison
        torch_tensor1 = torch.tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        torch_tensor2 = torch.tensor([[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]]).to(self.device)

        torch_expected = torch.matmul(torch_tensor1, torch_tensor2)

        # Comparing results
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_reshape_then_matmul(self):
        """
        Test reshaping a tensor followed by matrix multiplication: (tensor.reshape(shape) @ other_tensor)
        """
        pyflashlight_tensor = pyflashlight.Tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        new_shape = [2, 4]
        pyflashlight_reshaped = pyflashlight_tensor.reshape(new_shape)
        
        pyflashlight_result = pyflashlight_reshaped @ pyflashlight_tensor
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        torch_expected = torch_tensor.reshape(new_shape) @ torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_batched_matmul(self):
        """
        Test batched matrix multiplication: BxMxP = BxNxM @ BxMxP
        """
        B = 3  # Batch size

        # Creating batched tensors for pyflashlight
        pyflashlight_tensor1 = pyflashlight.Tensor([[[1, 2], [3, -4], [5, 6], [7, 8]] for _ in range(B)]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]).to(self.device)

        pyflashlight_result = pyflashlight_tensor1 @ pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        # Converting to PyTorch tensors for comparison
        torch_tensor1 = torch.tensor([[[1, 2], [3, -4], [5, 6], [7, 8]] for _ in range(B)]).to(self.device)
        torch_tensor2 = torch.tensor([[[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]).to(self.device)

        torch_expected = torch.matmul(torch_tensor1, torch_tensor2)

        # Comparing results
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_broadcasted_batched_matmul(self):
        """
        Test broadcasted batched matrix multiplication: BxMxP = NxM @ BxMxP
        """
        B = 3  # Batch size

        # Creating batched tensors for pyflashlight
        pyflashlight_tensor1 = pyflashlight.Tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]).to(self.device)

        pyflashlight_result = pyflashlight_tensor1 @ pyflashlight_tensor2
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        # Converting to PyTorch tensors for comparison
        torch_tensor1 = torch.tensor([[1, 2], [3, -4], [5, 6], [7, 8]]).to(self.device)
        torch_tensor2 = torch.tensor([[[2, 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]).to(self.device)

        torch_expected = torch.matmul(torch_tensor1, torch_tensor2)

        # Comparing results
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))



    def test_transpose_then_matmul(self):
        """
        Test transposing a tensor followed by matrix multiplication: (tensor.transpose(dim1, dim2) @ other_tensor)
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        dim1, dim2 = 0, 2
        pyflashlight_result = pyflashlight_tensor.transpose(dim1, dim2) @ pyflashlight_tensor
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch_tensor.transpose(dim1, dim2) @ torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_add_div_matmul_then_reshape(self):
        """
        Test a combination of operations: (tensor.sum() + other_tensor) / scalar @ another_tensor followed by reshape
        """
        pyflashlight_tensor1 = pyflashlight.Tensor([[[1., 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        scalar = 2
        new_shape = [2, 4]
        pyflashlight_result = ((pyflashlight_tensor1 + pyflashlight_tensor2) / scalar) @ pyflashlight_tensor1
        pyflashlight_result = pyflashlight_result.reshape(new_shape)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1., 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        torch_expected = ((torch_tensor1 + torch_tensor2) / scalar) @ torch_tensor1
        torch_expected = torch_expected.reshape(new_shape)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_scalar_power_tensor(self):
        """
        Test scalar power of a tensor: scalar ** tensor
        """
        scalar = 3
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = scalar ** pyflashlight_tensor
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = scalar ** torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_tensor_power_scalar(self):
        """
        Test tensor power of a scalar: tensor ** scalar
        """
        scalar = 3
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor ** scalar
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_expected = torch_tensor ** scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_tensor_sin(self):
        """
        Test sine function on tensor
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.sin()
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]).to(self.device)
        torch_expected = torch.sin(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_tensor_cos(self):
        """
        Test cosine function on tensor
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor.cos()
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor = torch.tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]).to(self.device)
        torch_expected = torch.cos(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_equal(self):
        """
        Test equal two tensors: tensor1.equal(tensor2)
        """
        pyflashlight_tensor1 = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 1], [7, 8]]]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor1.equal(pyflashlight_tensor2)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 1], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).to(self.device)
        torch_expected = (torch_tensor1 == torch_tensor2).float()

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_broadcasted_equal(self):
        """
        Test broadcasted equal two tensors: tensor1.equal(tensor2)
        """
        pyflashlight_tensor1 = pyflashlight.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[[10, 10]], [[5, 6]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor1.equal(pyflashlight_tensor2)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[10, 10]], [[5, 6]]]).to(self.device)
        torch_expected = (torch_tensor1 == torch_tensor2).float()

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        pyflashlight_tensor1 = pyflashlight.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)
        pyflashlight_tensor2 = pyflashlight.Tensor([[[10.0,], [-4.0,]],[[6.0,], [8.0,]]]).to(self.device)
        pyflashlight_result = pyflashlight_tensor1.equal(pyflashlight_tensor2)
        torch_result = utils.to_torch(pyflashlight_result).to(self.device)

        torch_tensor1 = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]).to(self.device)
        torch_tensor2 = torch.tensor([[[10.0,], [-4.0,]],[[6.0,], [8.0,]]]).to(self.device)
        torch_expected = (torch_tensor1 == torch_tensor2).float()

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))
    

    def test_zeros_like(self):
        """
        Test creating a tensor of zeros with the same shape as another tensor.
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_zeros = pyflashlight_tensor.zeros_like()
        torch_zeros_result = utils.to_torch(pyflashlight_zeros).to(self.device)

        torch_tensor_expected = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_zeros_expected = torch.zeros_like(torch_tensor_expected)

        self.assertTrue(utils.compare_torch(torch_zeros_result, torch_zeros_expected))

    def test_ones_like(self):
        """
        Test creating a tensor of ones with the same shape as another tensor.
        """
        pyflashlight_tensor = pyflashlight.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        pyflashlight_ones = pyflashlight_tensor.ones_like()
        torch_ones_result = utils.to_torch(pyflashlight_ones).to(self.device)

        torch_tensor_expected = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]).to(self.device)
        torch_ones_expected = torch.ones_like(torch_tensor_expected)

        self.assertTrue(utils.compare_torch(torch_ones_result, torch_ones_expected))


if __name__ == '__main__':
    unittest.main()
