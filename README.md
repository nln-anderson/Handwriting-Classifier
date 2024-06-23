# Handwriting Classifier - Classifying Handwritten Math Symbols
As we move into the future, more and more mathematical and scientific papers are created digitally. Personally, I used LaTeX and other coding languages to create organized write-ups for math content during undergraduate. I always wished there was an effective way to convert papers I wrote by hand into LaTeX documents. The goal of this project was to create such a program. The subproblem would be to create a program that can first classify individual handwritten symbols. This is what we will focus on for the time being. Do note that this readme file was written after completeting the program. There were many trials and errors along the way, from network models that wouldn't converge, to not properly preprocessing data. This write-up reflects the final process I used.
# Machine Learning Model for Symbol Recognition
The first step of this project was to create a machine learning model that can effectively classify characters. To do this, I used the following dataset from https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols. It consisted of 82 different characters, from greek letters to operations symbols. This is exactly what I was looking for.
# Data Transforms
Now, as with most data science and machine learning projects, the data wasn't perfect from the start. Yes, it consisted of most off the symbols I was looking for. However, the data was relatively uniform. Here is what I mean. Take a look at this image from the dataset <>.I transformed all the images to grayscale, since color is irrelevent in this situation. I also normalized the grayscale values to ensure stability within the gradient calculations.
# Viewing the Data
To get a feel for the data, I printed one example from each class. Take a look at the image.
(Image Here)
I noticed that the "times" and "x" class were almost indistinguishable. They both were x's. And so, I combined them into one class for the model. Similarly, "1" and "ascii_124" looked almost idential, so I combined them into one class as well, under the label "1." In terms of data balancing, I counted the number of instances of each class and saw that things were very imbalanced. I utilized Pytorch weighted sampling to ensure that each patch contained equal representation of each class.



