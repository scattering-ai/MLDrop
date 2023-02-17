# MLDrop client
Easily deploy PyTorch models for serving with a few lines.

[MLDrop](https://scattering-ai.webflow.io/) is a platform that allows you to deploy PyTorch models for serving.

You can use it for FREE. [Create an account](https://api.scattering.ai/signup?utm_source=landing).

## Installation
1) Install mldrop_client module:

```bash
pip install --upgrade mldrop_client
```

2) Create an account for FREE [here](https://api.scattering.ai/signup?utm_source=landing) and get the access token from your email.

## Usage
1) Load your PyTorch model and deploy it to MLDrop (see full example [here](examples/hello_world.py)):

```python
# Import MLDrop
import mldrop_client

# Import PyTorch
import torch

# Init mldrop (using your account token)
MLDROP_ACCESS_TOKEN = "USE_YOUR_ACCOUNT_TOKEN"
mldrop = mldrop_client.init(MLDROP_ACCESS_TOKEN)

# Your PyTorch model (dummy model that adds two numbers)
class MyPytorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

# Load your PyTorch model (place your model here)
model: torch.nn.Module = MyPytorchModel()

# Define model metadata: name + inputs + outputs
model_metadata = mldrop.create_model_metadata(
    model_name="my_hello_world_model", # Unique name for this model
    inputs=[
        mldrop.input_float("a"), # First input
        mldrop.input_float("b"), # Second input
    ],
    outputs=[
        mldrop.output_float("result"), # One single output
    ],
)

# Deploy model to MLDrop and get model_id
model_id = mldrop.deploy_pytorch_model(model_metadata, model)
# Done! Your model is up and running in the cloud ready to be consumed
```

2) Invoke model through Python API:
```python
# Use model_id to invoke model
# Invocation samples are defined as key-value dictionaries where keys should match the model expected inputs
samples = [
    {"a": 3, "b": 5},
    {"a": 7, "b": 2},
]
output = mldrop.invoke_model(model_id, samples)
# You get one output for each invocation sample, in the same order
print(output)
```

3) Invoke model through REST - GET
```bash
curl "https://api.scattering.ai/api/v1/model/invoke?t=USE_YOUR_ACCOUNT_TOKEN&model_id=DEPLOYED_MODEL_ID&a=3&b=5"
```
4) Invoke model through REST - POST
```bash
curl -X POST https://api.scattering.ai/api/v1/model/invoke -d '{
    "t":"USE_YOUR_ACCOUNT_TOKEN", 
    "model_id":"DEPLOYED_MODEL_ID", 
    "samples": [
        {"a": 3, "b": 5}, 
        {"a": 7, "b": 2}
    ]
}'
```







