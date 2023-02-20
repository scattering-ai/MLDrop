"""
Deploy a pre-trained ResNet-18 from torchvision to MLDrop.
Perform image classification for the given image URLs.
It also shows how to perform pre and post-processing inside the PyTorch model itself.

Installation:
pip install --upgrade mldrop_client
pip install torch
pip install torchvision
"""
import typing
import torch
import torchvision

# Import MLDrop
import mldrop_client


def main():
    """
    Deploy a PyTorch model to MLDrop and then invoke it
    """
    # Init MLDrop using your access token. Create an account in: https://www.scattering.ai/
    MLDROP_ACCESS_TOKEN = "USE_YOUR_ACCOUNT_TOKEN"
    mldrop = mldrop_client.init(MLDROP_ACCESS_TOKEN)

    # Load all image classification categories from ImageNet dataset
    imagenet_dict = load_imagenet_dict("data/imagenet_class_index.json")

    # Create model metadata
    model_metadata = mldrop.create_model_metadata(
        model_name="image_classification_resnet",
        description="Classify images with ResNet-18",
        inputs=[
            mldrop.input_image_url("img_url") # Input image sent as URL
        ],
        outputs=[
            # 3 outputs: class probability, class index and class name
            mldrop.output_float("class_prob"),
            mldrop.output_int("class_idx"),
            mldrop.output_string("class_name")
        ],
        sample_invocation=[
            {"img_url": "https://www.statbroadcast.com/grfx/icons/sogame.png"}
        ]
    )

    # Load pre-trained ResNet model
    pytorch_model = load_resnet_model_with_wrapper(imagenet_dict)

    # Deploy model to MLDrop
    model_id = mldrop.deploy_pytorch_model(model_metadata, pytorch_model)

    # Sample images URLs to use for invocation
    invocation_samples = [
        {"img_url": "https://www.androidfreeware.mobi/img2/com-devgames-flight-simulator-airplane.jpg"}, # Airplane
        {"img_url": "https://pbs.twimg.com/profile_images/1371605496340221955/IzD3qCAz_400x400.jpg"}, # Monkey
        {"img_url": "https://www.statbroadcast.com/grfx/icons/sogame.png"}, # Ball
    ]

    # Invoke model
    output = mldrop.invoke_model(model_id, invocation_samples)

    # Print output, one row per sample
    for out_row in output:
        print(f"Class: {out_row['class_name']}, idx: {out_row['class_idx']}, prob: {out_row['class_prob']}")


def load_resnet_model_with_wrapper(classes_dict: typing.Dict[int, str]) -> torch.nn.Module:
    """
    Load pre-trained ResNet18 image classification model
    """
    # Load TorchVision ResNet-18 pre-trained model
    resnet = torchvision.models.resnet18(pretrained=True)

    # Wrap with a model that performs pre and post-processing
    return ResNetWrapperModel(resnet, classes_dict)


class ResNetWrapperModel(torch.nn.Module):
    """
    Wrapper mopdel for ResNet that performs pre and post-processing inside the model and can be converted to TorchScript
    """
    def __init__(self, rest_net_module: torch.nn.Module, classes_dict: typing.Dict[int, str]):
        super().__init__()
        self.rest_net_module = rest_net_module
        self.classes_dict = classes_dict
        self.img_transform = torch.nn.Sequential(
            torchvision.transforms.Resize([256,]), # Resize image to 256
            torchvision.transforms.CenterCrop(224), # Crop
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize image
        )

    def forward(self, X: typing.List[torch.Tensor]):
        # X: list of images as Tensor (original images, before resize/norm)

        # Perform transformations to each input tensor
        X = [self.img_transform(img_tensor).unsqueeze(0) for img_tensor in X]
        # Concat all in a single batch-tensor
        X = torch.cat(X)

        # Invoke RestNet model
        Y = self.rest_net_module(X)

        # Process output
        Y = torch.nn.functional.softmax(Y, dim=1)
        out_idx = torch.argmax(Y, dim=1).flatten()

        # Classes lookout
        out_classes = [self.classes_dict.get(out_idx[i], "?") for i in range(len(out_idx))]
        return Y[:, 0], out_idx, out_classes


def load_imagenet_dict(path: str) -> typing.Dict[int,  str]:
    """
    Load dictionary of all ImageNet classes
    Key: class_index (as string)
    Value: [class_name, class_description]
    """
    import json
    with open(path) as json_file:
        d = json.load(json_file)
        out = {}
        for key, obj in d.items():
            out[int(key)] = obj[1]
        return out


if __name__ == "__main__":
    main()
