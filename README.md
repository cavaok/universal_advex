# Examining the Universal Properties of Adversarial Examples from a Dynamical Systems Perspective

## Overview
This repository contains the implementation and experimental framework used to investigate the universality of adversarial examples across different neural network architectures. The research challenges the widely accepted notion that adversarial examples are universal across model architectures by conducting a comprehensive comparative study between traditional MLPs and Dynamical Systems Architectures.
While existing literature suggests these adversarial examples are universal, our research presents evidence challenging this assumption.
## Key Concepts:
### TARGETED ADVERSARIAL EXAMPLES
Adversarial examples in machine learning involve strategically and iteratively modifying input data to cause confusion in pre-trained models. 
<img width="1070" alt="non-targeted" src="https://github.com/user-attachments/assets/6b42e7dd-4a41-4983-acb8-5fe5450c27a0" />
This visualization, however, is not impressive in the slightest– of course the model would be confused by the image of complete static! The interesting stuff happens when there is a **disconnect** between what **we perceive** and what the **model perceives**…
To illustrate this, I went ahead and trained an adversarial example against the popular HuggingFace model, MobileNetV2. Can you spot the adversarial example?
<img width="1176" alt="targeted" src="https://github.com/user-attachments/assets/e960c51f-7fed-481d-a63a-46ce6451d18a" />
Turns out, whichever one you guessed, you are correct. **All of these images** are adversarial examples! Not only that, but MobileNetV2 thinks these are all images of Shih Tzus with varying levels of certainty. 

### HOW IS THIS POSSIBLE?
While you might imagine these boundaries as lines neatly dividing categories, the reality is far more complex. Take Newton's Basins of Attraction for example:

<img width="176" alt="newtons" src="https://i.sstatic.net/0wHfa.jpg" />
This intricate pattern comes from just a single polynomial equation in two dimensions. Now imagine how mind bendingly complex the decision boundaries must be in a neural network, operating in high-dimensional space with layer upon layer of matrix multiplications and non-linear transformations... pretty crazy! We can conceptualize these wild decision boundaries as tentacle-like regions of each class stretching and interweaving into other classification spaces, creating vulnerable pockets where adversarial examples can hide.

### BIG QUESTION
This brings about the central questions: **Are adversarial examples universal? Or are they a function of the architecture?** Meaning, do some architectures help "fight back" these confusing, tentacle-like regions?

To answer this question, we conducted a comparative study of Multilayer Perceptrons (MLPs) vs Dynamical Systems Architectures. 

### DYNAMICAL SYSTEMS ARCHITECTURES
- Iterarive Denoising Autoencoder (see files marked auto for architecture)
- Iterative Hadamard Model (see files marked hadamard)

## Research HighlightsNon
- Trained and evaluated 200+ models with varying architectures --> Moved forward with 45 models
- Generated 40.5k adversarial examples (900 per model)
- Found that Dynamical Systems Architectures consistently outperformed traditional MLPs in terms of robustness

## Repository Structure
I separated the repository into three branches that show three distrinct phases in development/implementation: 
1. sweeps --> here I conducted sweeps on all the models I wanted to train
2. train --> here I trained all the models and saved their weights
3. current main branch --> all of my adversarial example training and logging

## Technologies Used
- Weights & Biases for experiment tracking
- Supabase for cloud-hosted metrics
- WPI Turing Supercomputing resources

## Results
Our research demonstrates that adversarial examples seem to be architecture-dependent, with some dynamical systems architectures showing superior robustness compared to traditional MLPs.

## Pending Citation
If you use this code or our findings in your research, please cite:
```
[Will be thesis paper citation...]
```

## License
This repository is currently unlicensed and under exclusive copyright. A license may be added at a later date.

## Contact
Email: okcava@wpi.edu

## Acknowledgments
- Advisor Dr. Randy Paffenroth
- WPI Turing Supercomputing facility

---
*This research is part of a Master's thesis at Worcester Polytechnic Institute*
