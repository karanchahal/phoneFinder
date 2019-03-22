# PhoneFinder
A Small Mask RCNN prototype that learns to detect the x,y coordinates of a phone in an image. This project contains the training and inference script.

### Setup
The project assumes you're running it on Google Colab which ha a lot of libs preinstalled. 
TO install those that colab doesnt have and to setup the project, run
```sh setup.sh```

### Training the Model
1. Run ```python train_phone_finder.py <path-to-dataset>```

### Running inference
```python find_phone.py <path-to-image>```

This project assumes that an image from the dataset is used to run inference on, specifically the image is 490 by 326 in height and width. If someone wants to submit a pull request to make the code image shape invariant, they're most welcome.

### Demo

I have detailed all the steps to setup and visualise the results in a Google Colab notebook [here](https://colab.research.google.com/drive/1elkvDtIvQmYwKKCN4-x-ZGJBA0j6Fulc)
