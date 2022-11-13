# Decision Boundary on the Training Samples
This is the code base for our decision boundary paper: The Vanishing Decision Boundary Complexity and the Strong First Component, which studies the decision boundary of deep neural networks on the training samples. Previous works rely on the adversarial samples for this purpose. 

## Why do we study decision boundary?
- Decision boundary helps understand the generalization of our classifier. In machine learning, it is well known that different levels of complexity in the decision bounary tell us the generalization ability of classifiers. For example, see this [illustration](https://en.wikipedia.org/wiki/Overfitting#/media/File:Overfitting.svg) for how overfitting contributes to the complexity in the decision boundary.

- Decision boundary is also useful in many other applications of deep learning, see the Related Work section in the paper for details.

## Steps for running our experiments:

1. Train a model (such as VGG19) using an optimizer. Below shows using the SGD optimizer and learning rate anealling/scheduling starting from 0.1 and decays according to a Cosine rule. This code can save models every epoch, for example, to facilitate analyze the training without rerun the training process. 

>`python main.py --model VGG19 --optimizer sgd --lr 0.1 --lr_mode schedule --saved_dir ./run1_VGG19`

2. Generate the embedding features on the training samples. This provides the feature space that we are going to analyze in the paper.  To generate for the above trained model(s), run 
 
>`python output_space.py --model VGG19 --saved_dir ./run1_VGG19` 
>Before this, run `extract_test_objects_classwise.py` to extract the test objects arranged in a dataloader by each class. 

3. Run `python generate_mean_and_variance.py` to generate the centers (for individual classes) and singulvar values (of the joint features matrix between a class pair) during training. This code forms the joint feature matrix between a given class pair (c1, c2), perform PCA and prodcues, e.g., the class center of class c1 and class c2. 

4. Run `python plot_cat_dog_in_PCA2.py` to plot the cat and dog samples in the PCA(2) space. This is a simpler version of Fig1, without the background plotting. This code just loads the class centers generated in Step 3 and plot. 

5. Run `python plot_fig1_fig2_fig3_fig6_fig9_fig10.py` to plot the figures. 

6. Run `python generate_fig7.py` to plot Fig 7. 
 
7. Run `plot_fig4_fig5_fig8.py` to produce the three figures. 

8. To make a video of the decision boundary evolution for a training process, run `python decision_boundary_evolution_video.py`






