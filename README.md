
# Vintage Colorizer

----------------------------

Image (artistic) [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/Team-16-B/VintageColorizer/blob/master/ImageColorizerColab.ipynb) |
Video [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/Team-16-B/VintageColorizer/blob//master/VideoColorizerColab.ipynb)
## The Technical Details

This is a deep learning based model.  More specifically, what I've done is combined the following approaches:

### [Self-Attention Generative Adversarial Network](https://arxiv.org/abs/1805.08318)

Except the generator is a **pretrained U-Net**, and I've just modified it to have the spectral normalization and self-attention.  It's a pretty straightforward translation.

### [Two Time-Scale Update Rule](https://arxiv.org/abs/1706.08500)

This is also very straightforward – it's just one to one generator/critic iterations and higher critic learning rate.
This is modified to incorporate a "threshold" critic loss that makes sure that the critic is "caught up" before moving on to generator training.
This is particularly useful for the "NoGAN" method described below.


Image [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/Team-16-B/VintageColorizer/blob//master/ImageColorizerColab.ipynb)
| Video [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/Team-16-B/VintageColorizer/blob//master/VideoColorizerColab.ipynb) 


### Your Own Machine


#### Hardware and Operating System Requirements

* **(Training Only) BEEFY Graphics card**.  I'd really like to have more memory than the 11 GB in my GeForce 1080TI (11GB).  You'll have a tough time with less.  The Generators and Critic are ridiculously large.  
* **(Colorization Alone) A decent graphics card**. Approximately 4GB+ memory video cards should be sufficient.
* **Linux**.  I'm using Ubuntu 18.04, and I know 16.04 works fine too.  **Windows is not supported and any issues brought up related to this will not be investigated.**

#### Easy Install

You should now be able to do a simple install with Anaconda. Here are the steps:

Open the command line and navigate to the root folder you wish to install.  Then type the following commands 

```console
git clone https://github.com/Team-16-B/VintageColorizer Vintage_Colorizer
cd Vintage_Colorizer
conda env create -f environment.yml
```

Then start running with these commands:

```console
source activate vintageColorizer
jupyter lab
```

From there you can start running the notebooks in Jupyter Lab, via the url they provide you in the console.  

> **Note:** You can also now do "conda activate vintageColorizer" if you have the latest version of conda and in fact that's now recommended. But a lot of people don't have that yet so I'm not going to make it the default instruction here yet.

### Installation Details

This project is built around the wonderful Fast.AI library.  Prereqs, in summary:

- **Fast.AI 1.0.51** (and its dependencies).  If you use any higher version you'll see grid artifacts in rendering and tensorboard will malfunction. So yeah...don't do that.
- **PyTorch 1.0.1** Not the latest version of PyTorch- that will not play nicely with the version of FastAI above.  Note however that the conda install of FastAI 1.0.51 grabs the latest PyTorch, which doesn't work.  This is patched over by our own conda install but fyi.
- **Jupyter Lab** `conda install -c conda-forge jupyterlab`
- **Tensorboard** (i.e. install Tensorflow) and **TensorboardX** (https://github.com/lanpa/tensorboardX).  I guess you don't *have* to but man, life is so much better with it.  FastAI now comes with built in support for this- you just  need to install the prereqs: `conda install -c anaconda tensorflow-gpu` and `pip install tensorboardX`
