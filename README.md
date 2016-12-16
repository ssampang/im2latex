# im2latex

This is a TensorFlow port of harvardnlp's [im2markup](https://github.com/harvardnlp/im2markup/) project. It contains the im2markup project as a git submodule to make use of its preprocessing scripts. 

## Run
To run, clone this repository, run the `prep.sh` (will install nodeJS if not already installed) script in the `data` directory, and then run `python im2latex.py`.

## Results
Unfortunately, as TensorFlow requires more memory than Torch, this project was not able to use the complete dataset used by the im2markup project. Assuming a batch size of 20, about one fourth of the dataset consisted of images that were too large to fit on a Titan X GPU. Still, over a period of 100 epochs, each taking approximately 100 minutes, the model seemed to saturate at 64.0% validation accuracy at around epoch 60. Test accuracy was only measured after 100 epochs, and was 64.9%. The im2markup project made use of beam search to measure test accuracy, which has not been implemented in this project yet. This, and the inability to use images from the full dataset, may help to explain the difference in performance relative to the im2markup project, which achieved 75% test accuracy on the full dataset.

## Future Work
I hope to look into implementing beam search, and running this model on AWS on a p2.8xlarge instance, where I can distribute the model across multiple GPUs.
