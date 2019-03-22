# Implementation Notes

Although this phone detector utilises a Mask RCNN as it's underlying object detector. I tried to use much simpler algorithms to detect the phone in the image first.

The various approaches are tried are as follows:

1. Divided the image into bins/grids and tried to classify each grid as a phone or background. I used a pretrained Resnet backbone
to get the image features and then I attached a 400 size dense layer along with the sigmoid activation to give phone/background predictions across 400 grid cells in the image.
I trained this with binary cross entropy loss but sadly due to the heavy class imbalance the model always predicted background.

2. I tried using a focal loss and a weighted binary cross entropy oss to solve this class imbalance problem but the model wasn't training/ learning 
anything. 

3. Lastly, I tried using K means on the image and tried dividing the image into 2 classes but due to the background noise, I couldn't isolate the phone and the floor well.

Finally, I turned to the Mask RCNN solution, by taking a pretrained network and just fintunning the heads of the network to predict the bounding box 
coordinates of the phone. After training successfully, on inference I took the predicted bounding box coordinates and found it's center, which I printed out
as the final result x,y coordinates. I was able to get a great accuracy for these and it detected phones in each of the validation set images
inside the 0.05 normalised radius as stipulated by the challenge. 

(I could have used the harr cascade approach by using Open CV but I found a good solution in my Mask RCNN hence didnt explore further.)

## Future Work

I would like to find out various metrics of the model to find how good the classifier is first. This would entail getting
1. Bias Variance curves.
2. Precision Recall curves
3. F1 Score. 
4. Augment Data and train on more examples to increase robustness. THen verify if the model did better

Once I have this, I would be accurately be able to tell how good the model is. To make this phone detector better:
1. I think a lighter object detector would be great. The SSD architecture boosted with the focal loss would theoretically give fast and accurate predictions. As the Mask RCNN is a 
2 stage detector it is quite heavy to run even in the cloud.
2. I would also request more data from the client in a variety of different lighting and background conditions
to make the detector more robust.
