# motor-classification
This notebook is based on the research done by Lun et. al. found in this paper: [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7522466/](url), as well as many others.

The purpose of the notebook is to use the Physionet MI-EEG dataset to train a convolutional neural network to classify motor-imagery events.

What that means simply, is that we have a dataset where subjects were instructed to imagine themselves closing their right or left fist, but not actually closing them. The experimenters used EEG machines to pick up the brain-wave activity of those events.

We can then use that EEG data to train a network to classify similar events in real-time. While this may seem trivial, the ability for a computer to recognize brain-wave events has wide-ranging uses for scenarios where someone is unable to interface with a phone or computer using their hands.

This is a pet project. I think that Brain-Computer Interfaces are fascinating, and while the idea of a chip in your brain is little more than a meme today, I believe some version of a BCI will become ubiquitious within our lifetimes.

Humor me as I explore the world of BCI, and learn a little about creating and training CNNs along the way. This is a work in progress, so if you come across this and see an error message below, please feel free to fix it for me =)
