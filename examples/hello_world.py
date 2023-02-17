"""
This example shows you how to deploy a PyTorch model to MLDrop and then invoke it.

Installation:
pip install --upgrade mldrop_client
pip install torch
"""

# PyTorch import
import torch

# Import for mldrop
import mldrop_client


def main():
    # Init mldrop
    # Create a FREE account in: https://scattering-ai.webflow.io/
    # Then use the access token you'll get in your email
    MLDROP_ACCESS_TOKEN = "USE_YOUR_ACCOUNT_TOKEN"
    mldrop = mldrop_client.init(MLDROP_ACCESS_TOKEN)

    # Your PyTorch model, in this case a trivial model that adds up two numbers
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

    # Instantiate PyTorch model
    model = MyModel()

    # Define model metadata: name + inputs + outputs
    model_metadata = mldrop.create_model_metadata(
        model_name="my_test_model_sum", # Unique name for your model
        description="My test model - Add two numbers", # Some decription
        inputs=[ # List of expected inputs
            mldrop.input_float("a"), # First input
            mldrop.input_float("b"), # Second input
        ],
        outputs=[ # List of expected outputs
            mldrop.output_float("result"), # One single output in this example
        ],
        sample_invocation=[ # Test samples to warmup model
            {"a": 3, "b": 5},
        ]
    )

    # Deploy model, you'll get a model_id that can later be used for invocation
    model_id = mldrop.deploy_pytorch_model(model_metadata, model)
    print(f"Model_id={model_id}")

    # Invoke model. Each same is defined as key-value dictionary, where keys should map the expected inputs of your model.
    samples = [
        {"a": 3, "b": 5},
        {"a": 7, "b": 2},
    ]
    output = mldrop.invoke_model(model_id, samples)
    # You get one output for each invocation sample, in the same order
    print(output)

    # Print samples and outputs together
    for sample, out in zip(samples, output):
        print(f" - Input: {sample} => Output: {out}")


if __name__ == "__main__":
    main()