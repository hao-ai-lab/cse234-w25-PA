from typing import Any, Dict, List
import torch


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = None, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = {} if attrs is None else attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[torch.Tensor]
            The input values of the given node.

        Returns
        -------
        output: torch.Tensor
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        result_node =  Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )
        if "shape" in node_A.attrs and "shape" in node_B.attrs:
            shape_A = node_A.attrs["shape"]
            shape_B = node_B.attrs["shape"]
            inferred_shape = self._broadcast_shape(shape_A, shape_B)
            result_node.attrs["shape"] = inferred_shape
        return result_node

    def _broadcast_shape(self, shape1: tuple, shape2: tuple) -> tuple:
        """
        根据 NumPy 的广播规则推断两个形状的广播结果。
        假设 shape1 和 shape2 都是 tuple 类型。
        """
        # 将两个 shape 反转，便于从最低维开始比较
        rev1 = list(shape1[::-1])
        rev2 = list(shape2[::-1])
        max_len = max(len(rev1), len(rev2))
        result_rev = []
        for i in range(max_len):
            dim1 = rev1[i] if i < len(rev1) else 1
            dim2 = rev2[i] if i < len(rev2) else 1
            if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
            result_rev.append(max(dim1, dim2))
        return tuple(result_rev[::-1])

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        result_node = Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )
        # 如果两个输入都有 "shape" 属性，则按照广播规则推断输出 shape
        if "shape" in node_A.attrs and "shape" in node_B.attrs:
            shape_A = node_A.attrs["shape"]
            shape_B = node_B.attrs["shape"]
            inferred_shape = self._broadcast_shape(shape_A, shape_B)
            result_node.attrs["shape"] = inferred_shape
        return result_node

    def _broadcast_shape(self, shape1: tuple, shape2: tuple) -> tuple:
        """
        根据 NumPy 广播规则推断两个形状的广播结果。
        例如： (4, 1, 3) 与 (1, 5, 3)  的广播结果为 (4, 5, 3)。
        """
        rev1 = list(shape1[::-1])
        rev2 = list(shape2[::-1])
        max_len = max(len(rev1), len(rev2))
        result_rev = []
        for i in range(max_len):
            dim1 = rev1[i] if i < len(rev1) else 1
            dim2 = rev2[i] if i < len(rev2) else 1
            if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
            result_rev.append(max(dim1, dim2))
        return tuple(result_rev[::-1])

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )
        # 自动传播：输出节点的 shape 与输入节点的 shape 一致
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] * node.attrs["constant"]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return [output_grad * node.attrs["constant"]]


class GreaterThanOp(Op):
    """Op to compare if node_A > node_B element-wise."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        result_node = Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}>{node_B.name})",
        )
        # 如果两个输入都有 "shape" 属性，则按照广播规则推断输出 shape
        if "shape" in node_A.attrs and "shape" in node_B.attrs:
            shape_A = node_A.attrs["shape"]
            shape_B = node_B.attrs["shape"]
            inferred_shape = self._broadcast_shape(shape_A, shape_B)
            result_node.attrs["shape"] = inferred_shape
        return result_node

    def _broadcast_shape(self, shape1: tuple, shape2: tuple) -> tuple:
        """
        根据 NumPy 广播规则推断两个形状的广播结果。
        例如：(4, 1, 3) 与 (1, 5, 3) 的广播结果为 (4, 5, 3)。
        """
        # 反转两个 shape 便于从最低维开始比较
        rev1 = list(shape1[::-1])
        rev2 = list(shape2[::-1])
        max_len = max(len(rev1), len(rev2))
        result_rev = []
        for i in range(max_len):
            dim1 = rev1[i] if i < len(rev1) else 1
            dim2 = rev2[i] if i < len(rev2) else 1
            if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
            result_rev.append(max(dim1, dim2))
        return tuple(result_rev[::-1])

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 2
        return (input_values[0] > input_values[1]).float()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]


class SubOp(Op):
    """Op to element-wise subtract two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        result_node = Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}-{node_B.name})",
        )
        # 如果两个输入都有 "shape" 属性，则按照广播规则推断输出 shape
        if "shape" in node_A.attrs and "shape" in node_B.attrs:
            shape_A = node_A.attrs["shape"]
            shape_B = node_B.attrs["shape"]
            inferred_shape = self._broadcast_shape(shape_A, shape_B)
            result_node.attrs["shape"] = inferred_shape
        return result_node

    def _broadcast_shape(self, shape1: tuple, shape2: tuple) -> tuple:
        """
        根据 NumPy 广播规则推断两个形状的广播结果。
        例如：(4, 1, 3) 与 (1, 5, 3) 的广播结果为 (4, 5, 3)。
        """
        # 将两个 shape 反转，从最后一维开始比较
        rev1 = list(shape1[::-1])
        rev2 = list(shape2[::-1])
        max_len = max(len(rev1), len(rev2))
        result_rev = []
        for i in range(max_len):
            dim1 = rev1[i] if i < len(rev1) else 1
            dim2 = rev2[i] if i < len(rev2) else 1
            if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
            result_rev.append(max(dim1, dim2))
        return tuple(result_rev[::-1])

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise subtraction of input values."""
        assert len(input_values) == 2
        return input_values[0] - input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of subtraction node, return partial adjoint to each input."""
        return [output_grad, mul_by_const(output_grad, -1)]


class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            name=f"ZerosLike({node_A.name})"
        )
        # 如果输入节点具有 "shape" 属性，则输出节点的 shape 也与之相同
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.zeros_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            name=f"OnesLike({node_A.name})"
        )
        # 自动传播：如果输入节点具有 "shape" 属性，则输出节点的 shape 与其相同
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.ones_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class SumOp(Op):
    """
    Op to compute sum along specified dimensions.

    Note: This is a reference implementation for SumOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        # 创建输出节点时将 dim 和 keepdim 写入 attrs 中
        result_node = Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Sum({node_A.name})",
        )
        # 自动传播：若输入节点有 "shape" 属性，则根据 dim 与 keepdim 推断输出 shape
        if "shape" in node_A.attrs:
            input_shape = node_A.attrs["shape"]
            # 处理负数索引：将负数转换为正数索引
            dims_normalized = tuple(d if d >= 0 else len(input_shape) + d for d in dim)
            if keepdim:
                # 保持原维度，只将指定维度置为 1
                output_shape = list(input_shape)
                for d in dims_normalized:
                    output_shape[d] = 1
                result_node.attrs["shape"] = tuple(output_shape)
            else:
                # 删除指定的维度
                output_shape = [s for i, s in enumerate(input_shape) if i not in dims_normalized]
                result_node.attrs["shape"] = tuple(output_shape)
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].sum(dim=node.attrs["dim"], keepdim=node.attrs["keepdim"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        dim = node.attrs['dim']
        keepdim = node.attrs["keepdim"]

        if keepdim:
            return [output_grad]
        else:
            reshape_grad = expand_as_3d(output_grad, node.inputs[0])
            return [reshape_grad]


class ExpandAsOp(Op):
    """Op to broadcast a tensor to the shape of another tensor.

    Note: This is a reference implementation for ExpandAsOp.
          If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        result_node = Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )
        # 自动传播：如果目标节点 (node_B) 具有 "shape" 属性，则输出节点的 shape 与之相同
        if "shape" in node_B.attrs:
            result_node.attrs["shape"] = node_B.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""
        return [sum_op(output_grad, dim=0), zeros_like(output_grad)]


class ExpandAsOp3d(Op):
    """Op to broadcast a tensor to the shape of another tensor.

    Note: This is a reference implementation for ExpandAsOp3d.
          If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        result_node = Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )
        # 自动传播：如果目标节点具有 "shape" 属性，则输出节点的 shape 与其相同
        if "shape" in node_B.attrs:
            result_node.attrs["shape"] = node_B.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        print('expand_op', input_tensor.shape, target_tensor.shape)
        return input_tensor.unsqueeze(1).expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""
        return [sum_op(output_grad, dim=(0, 1)), zeros_like(output_grad)]


class LogOp(Op):
    """Logarithm (natural log) operation."""

    def __call__(self, node_A: Node) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            name=f"Log({node_A.name})",
        )
        # 自动传播：若输入节点具有 "shape" 属性，则输出节点的 shape 与输入一致
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the natural logarithm of the input."""
        assert len(input_values) == 1, "Log operation requires one input."
        return torch.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the Log node, return the partial adjoint to the input."""
        input_node = node.inputs[0]
        return [output_grad / input_node]


class BroadcastOp(Op):
    def __call__(self, node_A: Node, input_shape: List[int], target_shape: List[int]) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            attrs={"input_shape": input_shape, "target_shape": target_shape},
            name=f"Broadcast({node_A.name}, {target_shape})",
        )
        # 自动传播：输出节点的 shape 直接设为 target_shape
        result_node.attrs["shape"] = tuple(target_shape)
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 1
        return input_values[0].expand(node.attrs["target_shape"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of broadcast node, return partial adjoint to input.

        For broadcasting, we need to sum out the broadcasted dimensions to get
        back to the original shape.
        """
        if "input_shape" not in node.attrs:
            raise ValueError("Input shape is not set. Make sure compute() is called before gradient()")

        input_shape = node.attrs["input_shape"]
        output_shape = node.attrs["target_shape"]

        dims_to_sum = []
        # 反向遍历各维度，根据广播规则确定需要求和的维度
        for i, (in_size, out_size) in enumerate(zip(input_shape[::-1], output_shape[::-1])):
            if in_size != out_size:
                dims_to_sum.append(len(output_shape) - 1 - i)

        grad = output_grad
        if dims_to_sum:
            grad = sum_op(grad, dim=dims_to_sum, keepdim=True)

        if len(output_shape) > len(input_shape):
            grad = sum_op(grad, dim=list(range(len(output_shape) - len(input_shape))), keepdim=False)

        return [grad]

class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        result_node = Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )
        # 如果两个输入都有 "shape" 属性，则按照广播规则推断输出 shape
        if "shape" in node_A.attrs and "shape" in node_B.attrs:
            shape_A = node_A.attrs["shape"]
            shape_B = node_B.attrs["shape"]
            inferred_shape = self._broadcast_shape(shape_A, shape_B)
            result_node.attrs["shape"] = inferred_shape
        return result_node

    def _broadcast_shape(self, shape1: tuple, shape2: tuple) -> tuple:
        """
        根据 NumPy 广播规则推断两个形状的广播结果。
        例如：(4, 1, 3) 与 (1, 5, 3) 的广播结果为 (4, 5, 3)。
        """
        rev1 = list(shape1[::-1])
        rev2 = list(shape2[::-1])
        max_len = max(len(rev1), len(rev2))
        result_rev = []
        for i in range(max_len):
            dim1 = rev1[i] if i < len(rev1) else 1
            dim2 = rev2[i] if i < len(rev2) else 1
            if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
            result_rev.append(max(dim1, dim2))
        return tuple(result_rev[::-1])

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        return input_values[0] / input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        x, y = node.inputs
        return [output_grad / y, mul_by_const(mul(output_grad, x), -1) / (y * y)]

class DivByConstOp(Op):
    """Op to element-wise divide a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )
        # 自动传播：如果输入节点具有 "shape" 属性，则输出节点的 shape 与输入一致
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] / node.attrs["constant"]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        return [output_grad / node.attrs["constant"]]


class TransposeOp(Op):
    """Op to transpose a matrix."""

    def __call__(self, node_A: Node, dim0: int, dim1: int) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            attrs={"dim0": dim0, "dim1": dim1},
            name=f"transpose({node_A.name}, {dim0}, {dim1})",
        )
        # 自动传播：如果输入节点有 "shape"，则输出的 shape 为输入 shape 交换指定维度后的结果
        if "shape" in node_A.attrs:
            input_shape = list(node_A.attrs["shape"])
            # 交换 dim0 和 dim1 对应的维度
            input_shape[dim0], input_shape[dim1] = input_shape[dim1], input_shape[dim0]
            result_node.attrs["shape"] = tuple(input_shape)
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the transpose of the input by swapping two dimensions.

        For example:
        - transpose(x, 1, 0) swaps first two dimensions.
        """
        assert len(input_values) == 1
        return input_values[0].transpose(node.attrs["dim0"], node.attrs["dim1"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of transpose node, return partial adjoint to input."""
        return [transpose(output_grad, node.attrs["dim1"], node.attrs["dim0"])]

class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        result_node = Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}@{node_B.name})",
        )
        if "shape" in node_A.attrs and "shape" in node_B.attrs:
            shape_A = node_A.attrs["shape"]  # 期望 (..., M, K)
            shape_B = node_B.attrs["shape"]  # 期望 (..., K, N)
            if len(shape_A) < 2 or len(shape_B) < 2:
                raise ValueError("MatMul requires both inputs to have at least 2 dimensions")
            M = shape_A[-2]
            K1 = shape_A[-1]
            K2 = shape_B[-2]
            N = shape_B[-1]
            if K1 != K2:
                raise ValueError(f"Inner dimensions do not match: {K1} vs {K2}")
            batch_A = shape_A[:-2]
            batch_B = shape_B[:-2]
            broadcast_batch = self._broadcast_batch(batch_A, batch_B)
            result_node.attrs["shape"] = broadcast_batch + (M, N)
        return result_node

    def _broadcast_batch(self, shape1: tuple, shape2: tuple) -> tuple:
        """
        根据 NumPy 广播规则对两个批量维度进行广播。
        如果 shape1 与 shape2 长度不同，则在左侧补 1。
        例如：(4, 3) 与 (1, 3, 5) 先补齐为 (1, 4, 3) 与 (1, 3, 5) 然后广播。
        """
        # 将较短的 shape 左侧补 1
        len1, len2 = len(shape1), len(shape2)
        if len1 < len2:
            shape1 = (1,) * (len2 - len1) + shape1
        elif len2 < len1:
            shape2 = (1,) * (len1 - len2) + shape2
        result = []
        for d1, d2 in zip(shape1, shape2):
            if d1 != d2 and d1 != 1 and d2 != 1:
                raise ValueError(f"Batch shapes {shape1} and {shape2} are not broadcastable")
            result.append(max(d1, d2))
        return tuple(result)

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the matrix multiplication result of input values."""
        assert len(input_values) == 2
        return torch.matmul(input_values[0], input_values[1])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input."""
        x, y = node.inputs
        return [matmul(output_grad, transpose(y, -2, -1)),
                matmul(transpose(x, -2, -1), output_grad)]



class SoftmaxOp(Op):
    """Softmax operation on input node."""

    def __call__(self, node_A: Node, dim: int = -1) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Softmax({node_A.name})",
        )
        # 自动传播：softmax 不改变输入形状，直接复制输入节点的 shape
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return softmax of input along specified dimension."""
        assert len(input_values) == 1
        exp_x = torch.exp(
            input_values[0] - torch.max(input_values[0], dim=node.attrs["dim"], keepdim=True).values
        )
        return exp_x / torch.sum(exp_x, dim=node.attrs["dim"], keepdim=True)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of softmax node, return partial adjoint to input."""
        softmax_out = softmax(node.inputs[0], dim=node.attrs["dim"])
        return [softmax_out * (output_grad - sum_op(output_grad * softmax_out, dim=(node.attrs["dim"],), keepdim=True))]



class LayerNormOp(Op):
    """Layer normalization operation."""

    def __call__(self, node_A: Node, normalized_shape: List[int], eps: float = 1e-5) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"LayerNorm({node_A.name})",
        )
        # 自动传播：LayerNorm 不改变形状，输出节点的 shape 与输入节点一致
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return layer normalized input."""
        assert len(input_values) == 1
        mean = torch.mean(input_values[0], dim=-1, keepdim=True)
        var = torch.mean((input_values[0] - mean) ** 2, dim=-1, keepdim=True)
        return (input_values[0] - mean) / torch.sqrt(var + node.attrs["eps"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Given gradient of the LayerNorm node wrt its output, return partial
        adjoint (gradient) wrt the input x.
        """
        x = node.inputs[0]
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]
        d = normalized_shape[-1]
        x_mean = sum_op(x, dim=(-1,), keepdim=True) / d
        x_var = sum_op(power(x - x_mean, 2), dim=(-1,), keepdim=True) / d
        inv_std = power(x_var + eps, -0.5)  # 1/sqrt(var+eps)
        g = output_grad
        g_mean = sum_op(g, dim=(-1,), keepdim=True) / d
        g_dot_xmu_mean = sum_op(g * (x - x_mean), dim=(-1,), keepdim=True) / d
        dx = inv_std * (g - g_mean - (x - x_mean) * power(inv_std, 2) * g_dot_xmu_mean)

        return [dx]


class ReLUOp(Op):
    """ReLU activation function."""

    def __call__(self, node_A: Node) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            name=f"ReLU({node_A.name})",
        )
        # 自动传播：ReLU 不改变形状，直接复制输入节点的 shape
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return ReLU of input."""
        assert len(input_values) == 1
        return torch.maximum(input_values[0], torch.tensor(0.0))

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of ReLU node, return partial adjoint to input."""
        return [output_grad * greater(node.inputs[0], mul_by_const(node.inputs[0], 0))]

class SqrtOp(Op):
    """Op to compute element-wise square root."""

    def __call__(self, node_A: Node) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            name=f"Sqrt({node_A.name})",
        )
        # 自动传播：如果输入节点有 "shape" 属性，则输出节点的 shape 与输入一致
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.sqrt(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [output_grad / (2 * sqrt(node.inputs[0]))]


class PowerOp(Op):
    """Op to compute element-wise power."""

    def __call__(self, node_A: Node, exponent: float) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            attrs={"exponent": exponent},
            name=f"Power({node_A.name}, {exponent})",
        )
        # 自动传播：如果输入节点有 "shape" 属性，则输出节点的 shape 与输入一致
        if "shape" in node_A.attrs:
            result_node.attrs["shape"] = node_A.attrs["shape"]
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0] ** node.attrs["exponent"]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [output_grad * node.attrs["exponent"] * power(node.inputs[0], node.attrs["exponent"] - 1)]


class MeanOp(Op):
    """Op to compute mean along specified dimensions.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        result_node = Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Mean({node_A.name})",
        )
        # 自动传播：如果输入节点具有 "shape" 属性，则推断输出节点的 shape
        if "shape" in node_A.attrs:
            input_shape = node_A.attrs["shape"]
            # 处理可能的负数索引，将它们转换为正数索引
            dims_normalized = tuple(d if d >= 0 else len(input_shape) + d for d in dim)
            if keepdim:
                out_shape = list(input_shape)
                for d in dims_normalized:
                    out_shape[d] = 1
                result_node.attrs["shape"] = tuple(out_shape)
            else:
                # 删除被归约的维度
                out_shape = [s for i, s in enumerate(input_shape) if i not in dims_normalized]
                result_node.attrs["shape"] = tuple(out_shape)
        return result_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.mean(input_values[0], dim=node.attrs["dim"], keepdim=node.attrs["keepdim"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        dims = node.attrs["dim"]  # dims 是一个 tuple
        input_shape = node.inputs[0].attrs["shape"]  # 假设输入节点已经有 shape 信息，例如 (50, 28, 128)
        scale = 1
        for d in dims:
            scale *= input_shape[d]
        return [output_grad / scale]

# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
softmax = SoftmaxOp()
layernorm = LayerNormOp()
relu = ReLUOp()
transpose = TransposeOp()
mean = MeanOp()
sum_op = SumOp()
sqrt = SqrtOp()
power = PowerOp()
greater = GreaterThanOp()
expand_as = ExpandAsOp()
expand_as_3d = ExpandAsOp3d()
log = LogOp()
sub = SubOp()
broadcast = BroadcastOp()

def topological_sort(nodes):
    """Helper function to perform topological sort on nodes.
    
    Parameters
    ----------
    nodes : List[Node] or Node
        Node(s) to sort
        
    Returns
    -------
    List[Node]
        Nodes in topological order
    """
    visited = set()
    sorted_list = []
    temp_mark = set()

    def visit(node):
        if node in temp_mark:
            raise ValueError("Cycle detected in computational graph!")
        if node not in visited:
            temp_mark.add(node)
            for input_node in node.inputs:
                visit(input_node)
            temp_mark.remove(node)
            visited.add(node)
            sorted_list.append(node)

    for node in nodes:
        visit(node)

    return sorted_list

class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, torch.Tensor]) -> List[torch.Tensor]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, torch.Tensor]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[torch.Tensor]
            The list of values for nodes in `eval_nodes` field.
        """
        # Step 1: Perform topological sorting to get the correct computation order
        sorted_nodes = self.topological_sort(self.eval_nodes)

        # Step 2: Initialize a dictionary to store computed values
        node_values = {}

        # Step 3: Assign given input values
        for node, value in input_values.items():
            node_values[node] = value

        # Step 4: Compute values for all nodes in topological order
        for node in sorted_nodes:
            if node in node_values:
                continue  # Skip if already assigned (input nodes)

            # Get input values for the node
            input_vals = [node_values[input_node] for input_node in node.inputs]

            # Compute the output value using the node's operation
            node_values[node] = node.op.compute(node, input_vals)

        # Step 5: Return the values of the requested evaluation nodes
        return [node_values[node] for node in self.eval_nodes]

    def topological_sort(self, nodes: List[Node]) -> List[Node]:
        """Perform topological sorting of the computational graph.

        Parameters
        ----------
        nodes: List[Node]
            List of nodes to be sorted.

        Returns
        -------
        sorted_nodes: List[Node]
            Nodes sorted in topological order.
        """
        visited = set()
        sorted_list = []
        temp_mark = set()

        def visit(node):
            if node in temp_mark:
                raise ValueError("Cycle detected in computational graph!")
            if node not in visited:
                temp_mark.add(node)
                for input_node in node.inputs:
                    visit(input_node)
                temp_mark.remove(node)
                visited.add(node)
                sorted_list.append(node)

        for node in nodes:
            visit(node)

        return sorted_list


def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input node respectively.
    """
    node_to_grad = {output_node: ones_like(output_node)}
    sorted_nodes = Evaluator([]).topological_sort([output_node])

    for node in reversed(sorted_nodes):
        if node in node_to_grad:
            output_grad = node_to_grad[node]

            # Skip computing gradient for Variable nodes (Placeholders)
            if isinstance(node.op, PlaceholderOp):
                continue

            input_grads = node.op.gradient(node, output_grad)

            for inp, grad in zip(node.inputs, input_grads):
                if inp in node_to_grad:
                    node_to_grad[inp] = node_to_grad[inp] + grad
                else:
                    node_to_grad[inp] = grad

    return [node_to_grad.get(node, zeros_like(node)) for node in nodes]

