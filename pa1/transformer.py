import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28


def linear_layer(X: ad.Node, W: ad.Node, b: ad.Node) -> ad.Node:
    """Implements a linear layer: output = X @ W + b."""
    output = ad.matmul(X, W) + b
    return output


def single_head_attention(Q: ad.Node, K: ad.Node, V: ad.Node, model_dim: int, model_dim_node: ad.Node) -> ad.Node:
    """Implements single-head attention mechanism."""
    # set the shape of the ad.Variable("model_dim") to be the square root of the model_dim
    scaling_factor = ad.sqrt(model_dim_node)
    attention_scores = ad.matmul(Q, ad.transpose(K, 2, 1)) / scaling_factor
    attention_weights = ad.softmax(attention_scores)
    return ad.matmul(attention_weights, V)

def encoder_layer(X: ad.Node, W_Q: ad.Node, W_K: ad.Node, W_V: ad.Node, W_O: ad.Node, W_1: ad.Node, W_2: ad.Node,
                  b_1: ad.Node, b_2: ad.Node, b_q: ad.Node, b_k: ad.Node, b_v: ad.Node, b_o: ad.Node, model_dim_const: ad.Node) -> ad.Node:
    """Implements a single encoder layer."""
    # check the dimensions of the input
    Q = linear_layer(X, W_Q, b_q)
    K = linear_layer(X, W_K, b_k)
    V = linear_layer(X, W_V, b_v)
    attention_output = single_head_attention(Q, K, V, X.shape[-1], model_dim_const)
    attention_output = linear_layer(attention_output, W_O, b_o)
    # print("encoder_layer: attention_output.shape =", attention_output.attrs.get("shape", "unknown"))
    # Feed-forward network
    ff_output = linear_layer(attention_output, W_1, b_1)
    ff_output = ad.relu(ff_output)
    ff_output = linear_layer(ff_output, W_2, b_2)
    # print("encoder_layer: ff_output.shape (after feed-forward) =", ff_output.attrs.get("shape", "unknown"))
    # ff_output = ad.layernorm(ff_output, normalized_shape=X.shape[-1])
    return ff_output


def transformer(X: ad.Node, nodes: List[ad.Node],
                model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """
    修改后 nodes 列表包含 15 个节点：
      [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2, b_q, b_k, b_v, b_o, model_dim_const, W_output, b_output]
    """
    # 解包节点列表
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2, b_q, b_k, b_v, b_o, model_dim_const, W_output, b_output = nodes

    # Transformer encoder layer
    encoder_out = encoder_layer(X, W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2, b_q, b_k, b_v, b_o, model_dim_const)

    # Average over the sequence length
    output = ad.mean(encoder_out, dim=(1,))  # (batch_size, model_dim)

    # Final classification layer
    output = linear_layer(output, W_output, b_output)  # (batch_size, num_classes)

    return output


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    log_probs = ad.log(ad.softmax(Z))
    total_loss = ad.mul_by_const(ad.sum_op(ad.mul(y_one_hot, log_probs), dim=(1,), keepdim=False), -1)
    average_loss = ad.sum_op(total_loss, dim=(0,), keepdim=False) / batch_size
    return average_loss




def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size> num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        # print(f"Batch {i}: X_batch.shape = {X_batch.shape}, y_batch.shape = {y_batch.shape}")
        # Compute forward and backward passes
        logits, loss_val, *grads = f_run_model(model_weights, X_batch, y_batch)
        # print(f"Batch {i}: logits.shape = {logits.shape}, loss = {loss_val.item()}")

        
        # Update weights and biases
        with torch.no_grad():
            for j in range(len(model_weights)):
                # grad_shape = grads[j].shape if grads[j] is not None else None
                # # print(f"  Weight {j} grad shape: {grad_shape}")
                model_weights[j] = (model_weights[j].detach() - lr * grads[j].sum(dim=0)).requires_grad_()
        # Hint: You can update the tensor using something like below:
        # W_Q -= lr * grad_W_Q.sum(dim=0)

        # Accumulate the loss
        total_loss += loss_val.item() * (end_idx - start_idx)


    # Compute the average loss
    
    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)

    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28
    seq_length = max_len  # 28
    num_classes = 10
    model_dim = 128
    eps = 1e-5
    num_epochs = 20
    batch_size = 25
    lr = 0.02

    # 创建输入及其他变量
    X = ad.Variable("X")
    W_Q = ad.Variable("W_Q")
    W_K = ad.Variable("W_K")
    W_V = ad.Variable("W_V")
    W_O = ad.Variable("W_O")
    W_1 = ad.Variable("W_1")
    W_2 = ad.Variable("W_2")
    b_1 = ad.Variable("b_1")
    b_2 = ad.Variable("b_2")
    b_q = ad.Variable("b_q")
    b_k = ad.Variable("b_k")
    b_v = ad.Variable("b_v")
    b_o = ad.Variable("b_o")
    model_dim_const = ad.Variable("model_dim")
    # 新增：创建最后一层参数变量
    W_output = ad.Variable("W_output")
    b_output = ad.Variable("b_output")

    # 设置 shape 属性
    X.attrs["shape"] = (batch_size, seq_length, input_dim)
    W_Q.attrs = {"shape": (input_dim, model_dim)}
    W_K.attrs = {"shape": (input_dim, model_dim)}
    W_V.attrs = {"shape": (input_dim, model_dim)}
    W_O.attrs = {"shape": (model_dim, model_dim)}
    W_1.attrs = {"shape": (model_dim, model_dim * 2)}
    W_2.attrs = {"shape": (model_dim * 2, model_dim)}
    b_1.attrs = {"shape": (model_dim * 2,)}
    b_2.attrs = {"shape": (model_dim,)}
    b_q.attrs["shape"] = (model_dim,)
    b_k.attrs["shape"] = (model_dim,)
    b_v.attrs["shape"] = (model_dim,)
    b_o.attrs["shape"] = (model_dim,)
    model_dim_const.attrs["shape"] = ()
    # 设置最后一层参数的 shape
    W_output.attrs["shape"] = (model_dim, num_classes)
    b_output.attrs["shape"] = (num_classes,)

    # 调用 transformer() 时将所有参数一起传入
    y_predict = transformer(
        X,
        [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2, b_q, b_k, b_v, b_o, model_dim_const, W_output, b_output],
        model_dim, seq_length, eps, batch_size, num_classes
    )

    y_groundtruth = ad.Variable("y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)

    # 在计算梯度时也包含 W_output 和 b_output
    grads = ad.gradients(loss, [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2, W_output, b_output])
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim * 2))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim * 2, model_dim))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim * 2,))
    b_2_val = np.random.uniform(-stdv, stdv, (model_dim,))
    W_output_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_output_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(model_weights, X_batch, y_batch):
        # print("f_run_model: X_batch.shape =", X_batch.shape)
        # print("f_run_model: y_batch.shape =", y_batch.shape)
        # print("f_run_model: W_Q.shape =", model_weights[0].shape)
        # print("f_run_model: W_K.shape =", model_weights[1].shape)
        # print("f_run_model: W_V.shape =", model_weights[2].shape)
        # print("f_run_model: W_O.shape =", model_weights[3].shape)
        # print("f_run_model: W_1.shape =", model_weights[4].shape)
        # print("f_run_model: W_2.shape =", model_weights[5].shape)
        # print("f_run_model: b_1.shape =", model_weights[6].shape)
        # print("f_run_model: b_2.shape =", model_weights[7].shape)
        # print("f_run_model: W_output.shape =", model_weights[8].shape)
        # print("f_run_model: b_output.shape =", model_weights[9].shape)
        result = evaluator.run({
            X: X_batch.clone().detach(),  # X_batch 的 shape 应该为 (batch_size, seq_length, input_dim)
            y_groundtruth: y_batch.clone().detach(),
            W_Q: model_weights[0],
            W_K: model_weights[1],
            W_V: model_weights[2],
            W_O: model_weights[3],
            W_1: model_weights[4],
            W_2: model_weights[5],
            b_1: model_weights[6],
            b_2: model_weights[7],
            W_output: model_weights[8],
            b_output: model_weights[9],
            b_q: torch.zeros(model_dim, dtype=torch.float64),
            b_k: torch.zeros(model_dim, dtype=torch.float64),
            b_v: torch.zeros(model_dim, dtype=torch.float64),
            b_o: torch.zeros(model_dim, dtype=torch.float64),
            model_dim_const: torch.tensor(model_dim, dtype=torch.float64)
        })
        # print("f_run_model: result[0].shape =", result[0].shape)
        # print("f_run_model: result[1].shape =", result[1].shape)
        return result

    # 修改后的 f_eval_model
    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size
        all_logits = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            # print(f"f_eval_model: Batch {i} X_batch.shape = {X_batch.shape}")
            logits = test_evaluator.run({
                X: X_batch.clone().detach(),
                W_Q: model_weights[0],
                W_K: model_weights[1],
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7],
                W_output: model_weights[8],
                b_output: model_weights[9],
                b_q: torch.zeros(model_dim, dtype=torch.float64),
                b_k: torch.zeros(model_dim, dtype=torch.float64),
                b_v: torch.zeros(model_dim, dtype=torch.float64),
                b_o: torch.zeros(model_dim, dtype=torch.float64),
                model_dim_const: torch.tensor(model_dim, dtype=torch.float64)
            })
            all_logits.append(logits[0].detach().numpy())
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [
        torch.tensor(W_Q_val,  dtype=torch.float64),
        torch.tensor(W_K_val,  dtype=torch.float64),
        torch.tensor(W_V_val,  dtype=torch.float64),
        torch.tensor(W_O_val,  dtype=torch.float64),
        torch.tensor(W_1_val,  dtype=torch.float64),
        torch.tensor(W_2_val,  dtype=torch.float64),
        torch.tensor(b_1_val,  dtype=torch.float64),
        torch.tensor(b_2_val,  dtype=torch.float64),
        torch.tensor(W_output_val,  dtype=torch.float64),
        torch.tensor(b_output_val,  dtype=torch.float64),
    ]


    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label == y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
