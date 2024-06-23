

<h1 align="center">Language Modeling</h1>
</div>

-- ReadMe is in progress --
## 1. Problem Statement
<p align="justify">
Language modeling is the task of predicting the next word in a sequence of text, given the previous words. It is a fundamental problem in natural language processing (NLP) with a wide range of applications. 
  Effective language models can be used to power intelligent assistants, improve machine translation, generate realistic text, enhance speech recognition, and even assist with code completion. 
  These models learn patterns in language by training on large text corpora, allowing them to capture the complexities of human communication. 
  Advancements in language modeling have been a driving force behind the recent progress in various NLP tasks, making it a crucial component in developing more natural and intelligent interactions
  between humans and machines.
</p>


## 2. Related Works
| Date | Title | Description | Code | Link |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| 2022 | Language Modeling with LSTMs in PyTorch | LSTM Model | | [Link](https://towardsdatascience.com/language-modeling-with-lstms-in-pytorch-381a26badcbf)
| 2020 | Mogrifier LSTM | Improved LSTM Architecture| [Code](https://github.com/google-deepmind/lamb)| [Link](https://paperswithcode.com/paper/mogrifier-lstm)
| 2018 | Regularizing and Optimizing LSTM Language Models | Improved LSTM Efficiency| [Code](https://github.com/salesforce/awd-lstm-lm)| [Link](https://paperswithcode.com/paper/regularizing-and-optimizing-lstm-language)
| 2024 | Build a Transformer-based language Model Using Torchtext | Transformer Model| | [Link](https://blog.paperspace.com/build-a-language-model-using-pytorch/)
|  | Training a causal language model from scratch | GPT2 Model| [Code](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section6_pt.ipynb) | [Link](https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch)

## 3. The Proposed Method
Given the computational constraints, an LSTM-based language model was chosen as the foundation. The proposed model incorporates several advanced regularization techniques inspired by the [work of Merity et al](https://paperswithcode.com/paper/regularizing-and-optimizing-lstm-language), including:

- Weight Tying: The input and output embedding weights are shared, reducing parameters and improving performance.
- Embedding Dropout: Dropout is applied to the input embeddings to prevent overfitting and improve generalization.
- Locked Dropout: The same dropout mask is applied across timesteps, encouraging the model to learn robust representations.
- Weight Drop: Dropout is applied to the recurrent weights of the LSTM, further stabilizing the learned representations.

By combining these techniques with the LSTM architecture, the goal is to create an efficient and high-performing model that can effectively leverage the available resources. The impact of these regularization methods on the model's effectiveness will be evaluated.

## 4. Implementation
In the following sections, the dataset, model architecture, and experimental results are described in detail.

### 4.1. Dataset
