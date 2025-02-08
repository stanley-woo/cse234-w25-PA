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
        mormalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]

        dims = tuple(range(-len(mormalized_shape), 0))
        mu = C.mean(dim=dims, keepdim=True)
        var = ((C - mu) ** 2).mean(dim=dims, keepdim=True)
        std = torch.sqrt(var + eps)
        res = (C - mu) / std

        return res

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        A = node.inputs[0]
        B = node.inputs[1]
        C = matmul(A, B)
        mormalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]
        dims = tuple(range(-len(mormalized_shape), 0))

        mu = mean(C, dim=dims, keepdim=True)
        powe = power(sub(C, mu), 2.0)
        var = mean(powe, dim=dims, keepdim=True)
        std = sqrt(add_by_const(var, eps))
        grad_mu = mean(output_grad, dim=dims, keepdim=True)
        grad_norm = mean(output_grad * sub(C, mu), dim=dims, keepdim=True)
        stuff = div(mul(sub(C, mu), grad_norm), mul(std, std))
        inv_std = div(ones_like(std), std)
        dc = mul(inv_std, (output_grad - grad_mu - stuff))
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
