# MLDrop client
Easily deploy PyTorch models for serving with a few lines.

[MLDrop](https://www.scattering.ai/) is a platform by [Scattering AI](https://www.scattering.ai/) that allows you to deploy PyTorch models for serving.

MLDrop client is the python lib that allows to deploy models to MLDrop.

## Features
- MLDrop is a serving platform specifically designed for [TorchScript](https://pytorch.org/docs/stable/jit.html) models.
- Easily deploy your [PyTorch](https://pytorch.org/) model with a few lines.
- Focus on your model, not DevOps, MLDrop takes care of launching instances, scaling, packaging, monitoring, etc.
- Your models can easily be invoked from anywhere using a simple REST API.
- MLDrop focus is efficient model inference, it doesn't add any constrain to your training pipeline. 

## How does MLDrop work
- Your model is converted to [TorchScript](https://pytorch.org/docs/stable/jit.html) and deployed for serving behind the scenes.
- The platform scales on demand as needed.
- Serving is performed using our hand-tuned inference server specifically optimized for TorchScript.

## Installation
1) Install mldrop_client module:

```bash
pip install --upgrade mldrop_client
```

2) [Create an account here](https://www.scattering.ai/signup?utm_source=github) and get the access token from your email.

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

# Your PyTorch model (in this example a dummy model that adds two numbers)
class MyPytorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

# Load your PyTorch model (place your model here)
model: torch.nn.Module = MyPytorchModel()

# Define some basic metadata: name + inputs + outputs
model_metadata = mldrop.create_model_metadata(
    model_name="my_hello_world_model", # Unique name for this model
    inputs=[
        mldrop.input_float("a"), # First input
        mldrop.input_float("b"), # Second input
    ],
    outputs=[
        mldrop.output_float("result"), # Mode output
    ],
)

# Deploy model to MLDrop and get model_id
model_id = mldrop.deploy_pytorch_model(model_metadata, model)
# Done! Your model is up and running in the cloud ready to be invoked
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

## Examples:
 - [Hello world - deploy a simple PyTorch](examples/hello_world.py)
 - [ResNet - Image classification using a pre-trained ResNet-18 model](examples/resnet.py)
 - [Roberta - text to embedding using a pre-trained Roberta model](examples/roberta_text_encoder.py)





