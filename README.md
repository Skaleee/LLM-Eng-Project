# LLM-Eng-Project

## Deepseek doc
1. clone:
`git clone https://github.com/deepseek-ai/DeepSeek-VL2`
2. create venv   
`cd DeepSeek-VL2/`  
`python -m venv .venv`  
activate venv in vscode or run:  
`.venv/Scripts/activate.bat`
3. install dependencies inside venv:  
Check https://pytorch.org/get-started/previous-versions/ for compatible versions for you. Torch *MUST* be `2.0.1`.   
`pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118`  
in requirements change xformers line to  
`xformers==0.0.21`  
then  
`pip install -r DeepSeek-VL2/requirements.txt`
4. optional: install deepseek as in project module:  
`pip install -e DeepSeek-VL2`
5. Replace the `inference.py` from the DeepSeek Repo with the one from this project and try running it. 