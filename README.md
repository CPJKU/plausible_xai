# Investigating the Plausibility of Explanations 

This repository accompanies the publication: 
Katharina Hoedt*, Verena Praher*, Arthur Flexer, and Gerhard Widmer, 
"Constructing Adversarial Examples to Investigate the Plausibility of Explanations in Deep Audio and Image Classifiers",
Neural Computing & Applications (2022). https://doi.org/10.1007/s00521-022-07918-7

(*) equal contribution.

This publication is an extension of our previous publication ([pdf](https://archives.ismir.net/ismir2021/paper/000066.pdf)) extending our experiments previously conducted in the audio domain to the image domain. In this repository we focus on the experiments in the image domain. 
For the experiments in the audio domain, check out 
[this repository](https://github.com/CPJKU/veracity).

This README is structured as follows:

* [Setup](#setup)
* [Note on Audio Data & Experiments](#audio)
* [Image Data & Models](#images)
* [Adversarial Attacks](#attacks)
* [Investigating Explainers](#explainers)

If you want to have a step-by-step description on how to reproduce the figures and tables
presented in the paper, check out [this secondary readme](REPRODUCTION.md), but make sure
to first check out the setup below.

If you have any issues running our code, please let us know! We will try our best 
to help you with any problems you might have.

## Setup <a name="setup"></a>
You can find required Python libraries in `requirements.txt`. If you want to use `conda`
and `pip`, you can run:

```
conda create -n py38-pxai python==3.8 -y
conda activate py38-pxai
pip install -r requirements.txt
```

to set up an environment with which you should be able to run our experiments. 

## Audio Data & Experiments <a name="audio"></a>
For the experiments on musical data we use the Jamendo dataset [1]. If you want to use this data as well, you can download the data 
[here](https://www.ismir.net/resources/datasets/) by choosing the dataset called *Jamendo-VAD*.

To get more information on the experiments on audio data and the audio model (sections 4.2, 5),
check out [this repository](https://github.com/CPJKU/veracity).


## Image Data & Models <a name="images"></a>

### Image Data
For image data experiments we use the ImageNet ILSVRC dataset [2]. You can obtain this
data by downloading it from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data).
The experiments are mostly performed on the validation set (containing 50,000 samples).


### Image Classifier(s)
For image classification, we use several pre-trained models available [here](https://pytorch.org/vision/stable/models.html).
You can compute the accuracy on the validation data for each pre-trained model
by using our script `prep/clean_accuracies.py`, e.g.

```
python -m prep.clean_accuracies --data-path <your-path-to-ImageNet> --arch alexnet
```

by defining the data-path as the path pointing to your ImageNet folder, and as a second
argument you can define one or multiple models you want to check (e.g. `alexnet`).

## Adversarial Attacks <a name="attacks"></a>

### Hyper-parameter search
For subsequent attacks, you can use the hyper-parameters we state in the paper. Depending on 
the architecture, you can either use the default arguments  of the script or you will 
have to adapt them accordingly. 

If you do not want to use our hyper-parameters but instead tune your own 
hyper-parameters for the adversarial PDG attack on ImageNet,
you can use the script in `adversarial/hyperparam_search.py`. You need to pass the path
pointing to the ImageNet data, and define the architecture of the model you want to attack, e.g.:

```
python -m prep.hyperparam_search --data-path <your-path-to-ImageNet> --arch alexnet
```

if you want to attack Alexnet. 

This script will store a pickle file in which you can compare
the number of adversaries that were found (out of a subset of 1000 validation samples) and
the accumulated difference between original and adversarial samples of different hyper-parameters.
You can also manually change the values of the hyper-parameters that you want to test. 

### ImageNet Attack

To compute the adversaries necessary for our experiments, you can use the 
`prep/compute_advs.py` script. You need to define the data-path pointing to ImageNet,
a save-path (to store the adversaries), the architecture of the model you want to attack and
attack parameters (self-tuned as described above or taken from the paper). 

To perform the attack on AlexNet with our default parameters, call

```
python -m prep.compute_advs --data-path <your-path-to-ImageNet>
```

or read upon the remaining command line arguments to adapt the hyper-parameters / goal model
accordingly.

### Computing Adversarial Accuracy

After an attack, you can compute the accuracy of a particular network given the adversaries
(or clean sample, if not available) as follows:

``` 
python -m prep.adv_accuracies --data-path <your-path-to-ImageNet> --adv-path <path-to-adversaries> --arch alexnet
```

You need to define the architecture for which adversaries were computed,
the path pointing to ImageNet, and the directory in which
you stored the adversaries of the defined architecture. 

## Investigating Explainers <a name="explainers"></a>

After you computed adversarial examples (for the architecture you are interested in), 
we can compute label-flip rates for different settings, and see whether a particular
explainer successfully detects potentially important parts of the input. 

To compute label-flip rates for a fixed k (k=3), as well as for varying k (see sections 6.2
and 6.3), you can run `experiments/label_flip_exps.py`. If, for example, you want to 
perform the experiment for AlexNet, LIME as an explainer and rectangular segments, run

```
python -m experiments.label_flip_exps --data-path <your-path-to-ImageNet> --arch alexnet --explainer lime 
```

This computes the label-flip rates for 10% of the ImageNet validation data.
Note that if you modified the standard path where you previously stored adversaries,
you will have to adapt the command line argument `adv-path`. 
If you prefer to run the experiments for SLIC segments instead of rectangles, first run 

```
python -m prep.compute_masks --data-path <your-path-to-ImageNet> --adv-path <your-path-to-adversaries> --arch alexnet
```

and then call `label_flip_exps` with the additional command line argument `--segment`. 

To perform the experiments with "localised" perturbations, where we investigate whether
an explainer can recover a fixed number of modified segments that changed a prediction, 
you can use `experiments/localised_attack.py`. For AlexNet and LIME as an Explainer, call

``` 
python -m experiments.localised_attack --data-path <your-path-to-ImageNet> --arch alexnet --explainer lime
``` 

If necessary, you will have to define the argument `--adv-path` again.
You can also add `--segment` here, if you prefer to run the experiment with SLIC segments
(if you computed them prior to that). 


## Citing
If you use this approach in your research, please cite the according paper:

```
@article{hoedt2022plausibility,
  title     = {Constructing adversarial examples to investigate the plausibility of explanations in deep audio and image classifiers},
  author    = {Hoedt, Katharina and Praher, Verena and Flexer, Arthur and Widmer, Gerhard},
  journal   = {Neural Computing and Applications},
  volume    = {35},
  number    = {14},
  pages     = {10011--10029},
  publisher = {Springer},
  year      = {2023}}
```

## License
Licensed under the [MIT License](LICENSE).

## References

[1] M.  Ramona,  G.  Richard,  and  B.  David,  “Vocal  Detection  in  Music  with  Support 
Vector  Machines,”  in Proc. of the IEEE Intern. Conference on Acoustics,  
Speech,  and  Signal  Processing,  ICASSP 2008,  Las  Vegas,  Nevada,  USA.
IEEE,  2008,  pp.1885–1888.

[2] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, 
A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg and L. Fei-Fei, “ImageNet Large 
Scale Visual Recognition Challenge”. International Journal of Computer Vision 115, 
211–252 (2015). https://doi.org/10.1007/s11263-015-0816-y

[3] J. Schlüter and B. Lehner,  “Zero-Mean Convolutions for Level-Invariant Singing 
Voice Detection,” in Proc. of the 19th Intern. Society for Music 
Information Retrieval Conference, ISMIR 2018, Paris, France, September 23-27, 2018, pp. 321–326.