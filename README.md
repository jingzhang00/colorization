# Diffusion Model and ControlNet Based Image
## Install
```bash
git clone git@github.com:jingzhang00/colorization.git
pip install -r requirements.txt
```

## How to use
```bash
python main.py
```
Then you will get two URLs, one for local and one for the public, open it in the browser then the webpage should look like the following:

**NOTE:** For negative prompt, we use the default setting "monochrome".

<img src="preview/webpage.png" height="75%" width="75%">

## Outcomes

### overview
<img src="preview/intro.png" height="30%" width="30%">

### compare with CNN/GAN based method
<img src="preview/compare.png" height="50%" width="50%">

### results when using different prompts
<img src="preview/prompt.png" height="50%" width="50%">