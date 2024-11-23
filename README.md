# llm-unichat
Universal Python API client for OpenAI, MistralAI, Anthropic, xAI, and Google AI.

## Build sequence:
```shell
rm -rf dist/*
```
```shell
python3 setup.py sdist bdist_wheel
```
```shell
twine upload dist/*         
```

## Usage:

1. Install the pip package:

```shell
pip install llm-unichat
```

2. Add the relevant functions to your application from module 'unichat':
- to initialise the API key: 'set_api_key'
- to generate a chat completion: 'get_chat_completion'
- [optional] import MODELS_LIST from 'models' for additional validation

## Functionality testing: 
Try the eclosed 'sample_chat.py' file:

```shell
python3 sample_chat.py
```