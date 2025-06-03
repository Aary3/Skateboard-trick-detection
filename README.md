# Skateboard-trick-detection
Classifying skateboard tricks in a given video.
Right now the program distingishes between ollie and kickflip.
# Requirements
> python 3.12 (for cuda) or newer versions (no cuda)
> tensorflow
> pandas
> numpy
> scikit-learn
> seaborn
> matplotlib
> keras
> cuda (for gpu computing)
# How to use it
Run the runner.py file using python and answer the given prompts to choose what you want to do. The basic usage to classify the trick on a given video is to answer 'video' and 'predict' in that order, then provide a link to the video. It can be a path to a video on your computer or a youtube link.
Best results when the video is short and shows only one trick.

This program was created as a individual project as a part of my studies at Warsaw University of Technology. I may add something to it in the future, but no guarantees.

# Citations
Part of the videos that I used to get the data to train the model come from this source, as it allows for academic usage:
https://github.com/LightningDrop/SkateboardML
