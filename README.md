# motor-classification
This project began in graduate school as a notebook based on the research done by Lun et. al. found in this paper: [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7522466/](url), as well as many others. I intended to build upon their research to create a more practical model for actual implementation in commercial Brain-Computer interface products. 

The purpose of the model is to use the Physionet MI-EEG dataset to train a convolutional neural network to classify motor-imagery events.

Simply, we have a dataset where subjects were instructed to imagine themselves closing their right or left fist, but not actually closing them. The experimenters used EEG machines to pick up the brain-wave activity of those events.

We can then use that EEG data to train a network to classify similar events in real-time. While this may seem trivial, the ability for a computer to recognize brain-wave events has wide-ranging uses for scenarios where someone is unable to interface with a phone or computer using their hands. They could simply imagine closing their fist, and trigger an event to interact with the computer.

I have since started fleshing out the notebook into a proper model to be trained via Sagemaker and deployed via ONNX for use on devices, and have refactored the previous Keras implementation to PyTorch. 

This is a pet project. I think that Brain-Computer Interfaces are fascinating, and while the idea of a chip in your brain is little more than a meme today, I believe some version of a BCI will become ubiquitious within our lifetimes.

Humor me as I explore the world of BCI, and learn a little about creating and training CNNs along the way. 
