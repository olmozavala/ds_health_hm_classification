#  Automatic detection of Pneumonia from Chest X-rays

In this homework we will solve a real-life problem using deep learning. 
The objective is to build a model that can automatically detect pneumonia from chest X-rays.

The dataset is available at [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

The dataset contains 5,863 X-ray images (JPEG) and 2 categories (Pneumonia/Normal). Because of the
large dataset I encourage you to test your code on a small subset of the data and then run it on HiperGator 
for the full dataset.

Specific objectives for this homework are:
1. Practice creating a PyTorch dataset
2. Practice creating a PyTorch data loader
3. Apply what we have learned about CNNs to solve a real-life problem
4. Practice using TensorBoard to visualize the training process

## Submission
You need to create a folder inside the `blue shared` folder of your team. There you will upload your data
and code.

### Download and visualize the data (10 pts)
Download the dataset from Kaggle and unzip it.
The dataset is available at [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

Remove any additional files and folders that you do not need. The dataset should contain only the following folders:
- `train` - training data with two subfolders `NORMAL` and `PNEUMONIA` 
- `test` - test data with two subfolders `NORMAL` and `PNEUMONIA`
- `val` - validation data with two subfolders `NORMAL` and `PNEUMONIA`

Make a notebook with a widget that allows the user to select a folder (`train`, `validation` or `test`) 
and will visualize *4 or more*  random images from each of the two classes (`NORMAL` and `PNEUMONIA`).

### Dataset (10 pts)
Create a PyTorch dataset **MyGenerator.py** that loads the data.
Inside your constructor you should receive at least the *input_folder* (`train`, `validation` or `test`) 
and the *transform* to apply to each of the samples.

*Tip: Remember that this dataset needs to return examples for both classes (NORMAL and PNEUMONIA). You can
use a threshold on the received **idx** from the data loader to decide which class to return.*

*Tip 2: The images are very different in size, intensity and contrast. You can use transformations like:
equalize, autocontrast, and resize to make the images more similar.*

### Model design (10 pts)
Please describe with words your proposed model architecture. 
Include information like which type of layers you will use, which activation functions to use
for the intermediate and output layers, and the loss function to use.

Feel free to include an image of the proposed architecture. 

One site for creating NN visualizations is [NN-SVG](http://alexlenail.me/NN-SVG/index.html).

### Model training and tensorboard (10 pts)
Following our example from class, create a **train.py** script that will train your model.
[Class Example](https://github.com/olmozavala/ISC_5935_EXamples/blob/main/Classification_MNIST/Training.py)

In you training script you should:
- Iterate by a number of epochs and batches, and evaluate the model on the validation set at the end of each epoch.
- Create a tensorboard writer and save the following information for each epoch:
    - Training loss
    - Validation loss
    - A sample of the input images, the ground truth labels, and the predicted labels
    - Additionally, save the model graph for the first epoch
- Keep track of the model with the lowest validation loss and save it to disk.

### Model evaluation (5 pts)
Create a **main.py** file or notebook that will have the general structure of a PyTorch project.
Here you will call your dataset, data loader, model, and training script.

### Describe results (10 pts)
Describe the results of your model. Include the following information:
- What was the best validation loss you got?
- What was the best accuracy you got?
- What was the best precision you got?
- What was the best recall you got?
- Show some of your results in tensorboard. Include a screenshot of the graph and the images. And discuss your results.
