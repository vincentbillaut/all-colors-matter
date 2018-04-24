# CS231n - Project Proposal

Vincent Billaut, Matthieu de Rochemonteix, Marc Thibault

Our goal is to tackle the problem of **colorization**, that
 is taking a gray scale photograph and translating it to its colorized 
version. This problem is particularly appealing to us, because it 
involves some kind of generative, creative work which is easily 
visualizable and even fun. Our implementation will feature GANs.  Several articles tackle this problem. One of them is [this one[insert proper citation here]](https://arxiv.org/abs/1603.08511).

One of the advantages of this project is that any dataset of colored images that is available can be used, since  we only have to generate the corresponding grayscale dataset. We will begin by using the [SUN  dataset](https://groups.csail.mit.edu/vision/SUN/), restricted to scenes of ocean, coast,beaches and lagoon. Depending on the first results, we will increase the variety of scenes included in the database and include human beings and faces to the dataset, and potentially  include other datasets if more variety is needed. 

We will begin by simply evaluating out results with the $\mathbb{L}^2$ norm on the RGB encoding of the image, but will keep in mind that this may not be the most relevant, and will investigate other color encodings such as *CIELAB* coordinates. 

One of the main advantages of this project is that the results will be very easy to validate manually, the overall coherence of the color being easily perceptible by the human eye.

If our first models perform well, we will try to extend the results to a wider variety of scenes and photographs and potentially to video sequence (the ideal end goal being to recolorize a grayscale video, which would need to introduce the notion of coherence between frames in a sequence). 

