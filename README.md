# Mulitclass Behavioral Classification

Multiclass behavioral classification is a project to automatically quantify specific behaviors in videos using MoViNet model architecture (https://arxiv.org/pdf/2103.11511). Currently designed to detect a single behavior with small adaptations needed to accomplish 'multiclass' goals. This project was designed with intent of determine duration, bouts, and timestamps for mouse behaviors in videos from experiments evaluating behavioral phenotypes.

## Usage

Multiple modules for training, prediction, and evaluation of new models

### dataset_generation
Create a tensorflow dataset for training, some overlap with training_models

### model_loading
Save and load models in standardized way. Current code has many parameters for data processing (batch size, sample rate, frame size), this saves all the relevant information with the reference to the weights that should be loaded to use the model

### prediction
Predict labels for a new video or set of videos. Reference the template to save video metadata correctly

### pull_annotations
Use VIA annotations to generate new clips to augment the training set. VIA annotator can be found here: https://www.robots.ox.ac.uk/~vgg/software/via/app/via_video_annotator.html

### training_models
Train a new model using previous weights or pretrained, publically available MoViNet weights

### video_preprocessing
Process specific videos used for this purpose. Automatically detect mouse 'arenas' from first video frame using OpenCV to crop videos down to a single animal. Crop and pad the videos to center animals of interest.


## License
MIT License

Copyright (c) 2025 Rebecca Krall

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.