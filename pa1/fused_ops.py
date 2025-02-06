from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps}, 
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        A = input_values[0]
        B = input_values[1]
        C = A @ B
        print(f"MatMulLayerNorm: C shape before normalized: {C.shape}")
        mormalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]

        C = C[:mormalized_shape[0]]
        print(f"MatMulLayerNorm: C shape after normalized: {C.shape}")

        mu = mean(C, dim=(-1,), keepdim=True)
        mu_sqared = mean(power(C, 2.0), dim=(-1,), keepdim=True)
        var = sub(mu_sqared, power(mu, 2.0))
        std = sqrt(var + eps)

        res = div(sub(C - mu), std)

        node.attrs["C"] = C
        node.attrs["mu"] = mu
        node.attrs["std"] = std

        return res

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        C = node.attrs["C"]
        mu = node.attrs["mu"]
        std = node.attrs["std"]

        normalized_C = div(sub(C - mu), std)
        grad_mu = mean(output_grad, dim=(-1,), keepdim=True)
        grad_norm = mean(mul(output_grad, normalized_C), dim=(-1,), keepdim=True)
        temp = sub(output_grad, grad_mu)
        dc = div(sub(temp, mul(normalized_C, grad_norm)), std)
        A = node.inputs[0]
        B = node.inputs[1]
        da = matmul(dc, transpose(B, -2, -1))
        db = matmul(transpose(A, -2, -1), dc)
        return [da, db]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        raise NotImplementedError

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
