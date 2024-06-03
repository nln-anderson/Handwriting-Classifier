# Handwriting-to-Print
As we move into the future, more and more mathematical and scientific papers are created digitally. Personally, I used LaTeX and other coding languages to create organized write-ups for math content during undergraduate. I always wished there was an effective way to convert papers I wrote by hand into LaTeX documents. The goal of this project is the create such a program.
# Machine Learning Model for Symbol Recognition
The first step of this project is to create a machine learning model that can effectively classify characters. To do this, I used the following dataset from https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols. It consists of 82 different characters, from greek letters to operations symbols.
# Data Transforms
I transformed all the images to grayscale, since color is irrelevent in this situation. I also normalized the grayscale values to ensure stability within the gradient calculations. 

