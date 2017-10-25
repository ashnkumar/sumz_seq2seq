# Sumz
![](https://img.shields.io/badge/python-3-brightgreen.svg) ![](https://img.shields.io/badge/tensorflow-1.1.0-orange.svg)

The model code for the chrome extension [(Sumz)](https://chrome.google.com/webstore/detail/sumz/odpjlfcmpnebjjjgdobgbjnbcfdlicgk) that implements a sequence to sequence model for summarizing Amazon reviews, using Tensorflow 1.1 and the <b>[Amazon Fine Foods reviews dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews).</b>

![Preview](https://github.com/ashnkumar/sumz_seq2seq/blob/master/images/sumz9.gif)

The `seq2seq_model_building.ipynb` notebook walks through building and training a [Sequence to sequence model](https://www.tensorflow.org/tutorials/seq2seq) with Tensorflow (version 1.1).

The model is currently used as the predictive backend for the <b>[Sumz](https://chrome.google.com/webstore/detail/sumz/odpjlfcmpnebjjjgdobgbjnbcfdlicgk)</b> chrome extension, which takes in Amazon reviews on the current web page and displays a small summary of each review. The model is trained on the the [Amazon fine food reviews dataset.](https://www.kaggle.com/snap/amazon-fine-food-reviews) from Kaggle, which consists of 568K review-summary pairs.

This builds on the [Text Summarization](https://github.com/Currie32/Text-Summarization-with-Amazon-Reviews) project by David Currie (this [Medium post](https://medium.com/towards-data-science/text-summarization-with-amazon-reviews-41801c2210b) goes into excellent detail as well).


## The Model

<p align="center">
<img src="https://github.com/ashnkumar/sumz_seq2seq/blob/master/images/nct-seq2seq.png"/>
</p>

<i>seq2seq model. source: [WildML](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/)</i>

Sequence-to-sequence models use two different RNNs, connected through the output state of the initial RNN. This is also called the encoder-decoder model (similar to Autoencoders). These seq2seq models are extremely powerful and versatile; they've been shown to have incredible performance a range of tasks including:

<p align="center">

| Task        | Input | Output
|:------------- |:------------- | :--------
| <b>Language translation</b>      | Text in language 1 | Text in language 2
| <b>News headlines</b> | Text of news article | Short headline
| <b>Question/Answering | Questions about content | Answers to questions
| <b>Chatbots</b> | Incoming chat to bot | Reply from chatbot
| <b>Smart email replies</b> | Email content | Reply to email
| <b>Image captioning</b> |Image | Caption describing image
| <b>Speech to text<b/> | Raw audio | Text of audio


For more information, here are some great resources:

* [Practical seq2seq](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/)
* [Tensorflow seq2seq tutorials](https://github.com/ematvey/tensorflow-seq2seq-tutorials)
* [Google talk by Quoc Le](https://www.youtube.com/watch?v=G5RY_SUJih4)
* [Deep Learning for Chatbots](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/)

We're using it here to 'translate' from a sequence of words (the entirety of an Amazon review) and another sequence of words (the short summary of the review).


## Notebooks

The two notebooks (`data_preprocessing.ipynb` and `seq2seq_model_building.ipynb`) walk through the following steps in building the end-to-end system:

* <b>Preprocessing the data</b>: Exploring the Amazon reviews dataset, converting reviews strings into integer vectors, then building a word embeddings matrix from the vocabulary
* <b>Building the model</b>: Building the sequence-to-sequence model layer by layer using Tensorflow
* <b>Training / testing the model</b>: Feeding the preprocessed data into the model, and generating our own summaries to check out the model's inference performance
* <b>Exporting the model for inference serving</b>: Converting the model into a serialized Protobuff format to serve in a production environment (the chrome extension)

## Credits
* Much credit to the [Text-Summarization-with-Amazon-Review](https://github.com/Currie32/Text-Summarization-with-Amazon-Reviews) by [Currie32](https://github.com/Currie32)
* The site [WildML](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) for extraordinarily helpful introductions to many of the key concepts in in the model


## License

### The MIT License (MIT)

Copyright (c) 2017 Ashwin Kumar<ash.nkumar@gmail.com@gmail.com>

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
> THE SOFTWARE.
