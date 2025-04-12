# unichat
Universal API chat Python client for OpenAI, MistralAI, Anthropic, xAI, Google AI, DeepSeek, Alibaba and Inception.

## Build sequence:
```shell
rm -rf dist build *.egg-info
```
```shell
python3 -m build
```
```shell
twine upload dist/*
```

## Usage:

1. Install the pip package:

```shell
pip install unichat
```

2. Add the class 'UnifiedChatApi' from module 'unichat' to your application:

3. [optional] Import MODELS_LIST as well for additional validation

## Functionality testing:
Try the eclosed in the source code 'sample_chat.py' file:

```shell
python3 sample_chat.py
```
