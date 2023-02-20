"""
This example shows how to create a Bert as a service model, using a pre-trained Roberta encoder.
For each input text it returns an embedding vector of length 768 that can later be used for other tasks, like text classification or KNN.

Installation:
pip install --upgrade mldrop_client
pip install torch
pip install torchtext
"""

# PyTorch import
import torch
import torchtext
import typing

# Import for mldrop
import mldrop_client


class RobertaEncoderWrapperMode(torch.nn.Module):
    """
    Wrapper for pre-trained XLM-Roberta encoder
    It converts a list of input text into a list of embedding vectors of length 768
    The model receives a list of strings as input and performs pre-processing (tokenization, etc) inside the model itself.
    It also does post-processing computing the average of the embeddings of all tokens.
    """
    def __init__(self):
        super().__init__()
        self.padding_idx = 1
        # Define set of transformations to tokenize the input text
        tokenizer_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
        vocab_url = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
        self.text_transform = torchtext.transforms.Sequential(
            torchtext.transforms.SentencePieceTokenizer(tokenizer_model_path), # Tokenize input text
            torchtext.transforms.VocabTransform(torch.hub.load_state_dict_from_url(vocab_url)), # Convert token to id based on vocabulary
            torchtext.transforms.Truncate(256 - 2), # Truncate to max length
            torchtext.transforms.AddToken(token=0, begin=True), # Add being token
            torchtext.transforms.AddToken(token=2, begin=False), # Add end token
        )
        self.encoder = torchtext.models.XLMR_BASE_ENCODER.get_model() # Load pre-trained XLM-Roberta model to encode input tokens into embeddings

    def forward(self, text_list: typing.Any) -> torch.Tensor:
        """
        Encode input text into embeddings.
        :param text_list: a list of text, of type List[str] but we need to use Any to make it work with TorchScript compiler
        :return: one 768 length embedding for each input text
        """
        # Pre-processing: transform input text into tokens
        tokens_tensor = torchtext.functional.to_tensor(self.text_transform(text_list), padding_value=self.padding_idx)
        # Encode tokens into embeddings
        encoded_tokens = self.encoder(tokens_tensor)
        # Compute average embedding across all tokens of each input text
        return torch.mean(encoded_tokens, 1)


def main():
    # Init mldrop
    # Create an account in: https://www.scattering.ai/
    # Then use the access token you'll get in your email
    MLDROP_ACCESS_TOKEN = "USE_YOUR_ACCOUNT_TOKEN"
    mldrop = mldrop_client.init(MLDROP_ACCESS_TOKEN)

    # Instantiate PyTorch model
    model = RobertaEncoderWrapperMode()

    # Define model metadata
    model_metadata = mldrop.create_model_metadata(
        model_name="roberta_encoder",
        description="Roberta text to embedding encoder",
        inputs=[
            mldrop.input_string("text_list"), # Input text
        ],
        outputs=[
            mldrop.output_float_list("result"), # One embedding vector of size 768 for each input text
        ],
        sample_invocation=[ # Test samples to warmup model
            {"text_list": "some text"},
        ]
    )

    # Deploy model, you'll get a model_id that can later be used for invocation
    model_id = mldrop.deploy_pytorch_model(model_metadata, model)
    print(f"Model_id={model_id}")

    # Invoke model
    samples = [
        {"text_list": "hello world"},
        {"text_list": "pytorch rocks"},
    ]
    output = mldrop.invoke_model(model_id, samples)

    # Print samples and outputs together
    for sample, out in zip(samples, output):
        print(f" - Input: {sample} => Output: ({len(out)}) {out}")


if __name__ == "__main__":
    main()
