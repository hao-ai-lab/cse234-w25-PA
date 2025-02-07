import torch
import pytest
from typing import Dict, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad
from fused_ops import matmul_layernorm, matmul_softmax


def check_compute_output(
    node: ad.Node, input_values: List[torch.Tensor], expected_output: torch.Tensor
) -> None:
    output = node.op.compute(node, input_values)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=1e-4)


def check_evaluator_output(
    evaluator: ad.Evaluator,
    input_values: Dict[ad.Node, torch.Tensor],
    expected_outputs: List[torch.Tensor],
) -> None:
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        torch.testing.assert_close(output_val, expected_val, atol=1e-4, rtol=1e-4)


def test_matmul_layernorm_forward():
    """Test forward pass of fused matmul + layernorm against separate ops."""
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    
    # Create computation graphs for both fused and separate ops
    fused_y = matmul_layernorm(x1, x2, normalized_shape=[4])
    separate_y = ad.layernorm(
        ad.matmul(x1, x2),
        normalized_shape=[4]
    )

    # Test input values
    x1_val = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0, 10.0], [10.0, 11.0, 12.0, 13.0], [13.0, 14.0, 15.0, 16.0]])

    # Create evaluator for separate ops
    evaluator = ad.Evaluator([separate_y])
    expected_output = evaluator.run({x1: x1_val, x2: x2_val})[0]

    # Check if fused op matches
    check_compute_output(fused_y, [x1_val, x2_val], expected_output)


def test_matmul_layernorm_backward():
    """Test backward pass of fused matmul + layernorm against separate ops."""
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    
    # Create computation graphs for both fused and separate ops
    fused_y = matmul_layernorm(x1, x2, normalized_shape=[4])
    separate_y = ad.layernorm(
        ad.matmul(x1, x2),
        normalized_shape=[4]
    )
    
    y_grad = ad.Variable("y_grad")
    
    # Get gradients from both approaches
    fused_x1_grad, fused_x2_grad = fused_y.op.gradient(fused_y, y_grad)
    separate_x1_grad, separate_x2_grad = ad.gradients(separate_y, [x1, x2])
    
    # Test input values
    x1_val = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0, 10.0], [10.0, 11.0, 12.0, 13.0], [13.0, 14.0, 15.0, 16.0]])
    y_grad_val = torch.ones((3, 4), dtype=torch.float32)

    # Create evaluators
    fused_evaluator = ad.Evaluator([fused_x1_grad, fused_x2_grad])
    separate_evaluator = ad.Evaluator([separate_x1_grad, separate_x2_grad])

    # Get expected gradients from separate ops
    expected_grads = separate_evaluator.run({
        x1: x1_val, 
        x2: x2_val, 
        y_grad: y_grad_val
    })

    # Check if fused op gradients match
    check_evaluator_output(
        fused_evaluator,
        input_values={x1: x1_val, x2: x2_val, y_grad: y_grad_val},
        expected_outputs=expected_grads
    )


def test_matmul_softmax_forward():
    """Test forward pass of fused matmul + softmax against separate ops."""
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    
    # Create computation graphs for both fused and separate ops
    fused_y = matmul_softmax(x1, x2, dim=-1)
    separate_y = ad.softmax(
        ad.matmul(x1, x2),
        dim=-1
    )

    # Test input values
    x1_val = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0, 10.0], [10.0, 11.0, 12.0, 13.0], [13.0, 14.0, 15.0, 16.0]])

    # Create evaluator for separate ops
    evaluator = ad.Evaluator([separate_y])
    expected_output = evaluator.run({x1: x1_val, x2: x2_val})[0]

    # Check if fused op matches
    check_compute_output(fused_y, [x1_val, x2_val], expected_output)


def test_matmul_softmax_backward():
    """Test backward pass of fused matmul + softmax against separate ops."""
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    
    # Create computation graphs for both fused and separate ops
    fused_y = matmul_softmax(x1, x2, dim=-1)
    separate_y = ad.softmax(
        ad.matmul(x1, x2),
        dim=-1
    )
    
    y_grad = ad.Variable("y_grad")
    
    # Get gradients from both approaches
    fused_x1_grad, fused_x2_grad = fused_y.op.gradient(fused_y, y_grad)
    separate_x1_grad, separate_x2_grad = ad.gradients(separate_y, [x1, x2])
    
    # Test input values
    x1_val = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0, 10.0], [10.0, 11.0, 12.0, 13.0], [13.0, 14.0, 15.0, 16.0]])
    y_grad_val = torch.ones((3, 4), dtype=torch.float32)

    # Create evaluators
    fused_evaluator = ad.Evaluator([fused_x1_grad, fused_x2_grad])
    separate_evaluator = ad.Evaluator([separate_x1_grad, separate_x2_grad])

    # Get expected gradients from separate ops
    expected_grads = separate_evaluator.run({
        x1: x1_val, 
        x2: x2_val, 
        y_grad: y_grad_val
    })

    # Check if fused op gradients match
    check_evaluator_output(
        fused_evaluator,
        input_values={x1: x1_val, x2: x2_val, y_grad: y_grad_val},
        expected_outputs=expected_grads
    )


if __name__ == "__main__":
    # Test forward pass
    test_matmul_layernorm_forward()
    test_matmul_softmax_forward()

    # Test backward pass
    test_matmul_layernorm_backward()
    test_matmul_softmax_backward()
