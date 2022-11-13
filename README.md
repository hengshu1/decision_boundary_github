# Decision Boundary on the Training Samples
This is the code base for our decision boundary paper: The Vanishing Decision Boundary Complexity and the Strong First Component, which studies the decision boundary of deep neural networks on the training samples. Previous works rely on the adversarial samples for this purpose. 

## Why do we study decision boundary?
- Decision boundary helps understand the generalization of our classifier. In machine learning, it is well known that different levels of complexity in the decision bounary tell us the generalization ability of classifiers. For example, see this [illustration](https://en.wikipedia.org/wiki/Overfitting#/media/File:Overfitting.svg) for how overfitting contributes to the complexity in the decision boundary.

- Decision boundary is also useful in many other applications of deep learning, see the Related Work section in the paper for details.

# Steps for running our experiments:

1. Train a model (such as VGG19) using an optimizer. Below shows using the SGD optimizer and learning rate anealling/scheduling starting from 0.1 and decays according to a Cosine rule. 

`python main.py --model VGG19 --optimizer sgd --lr 0.1 --lr_mode schedule --saved_dir ./run1`

2. Plot CCTM: Cross-Class Test accuracy Matrix/Map. 
   1. `python generate_cctm.py` This is used to generate the data for Figure 1 and Table 1 in the paper. 
   2. After this, run `python plot_cctm.py` to generate Figure 1. 

3. Generate the embedding features on the training samples. To generate for the above trained model(s), run 
 
`python output_space.py --model VGG19 --saved_dir ./run1`










