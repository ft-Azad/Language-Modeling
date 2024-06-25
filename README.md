

<h1 align="center">Language Modeling</h1>
</div>

-- ReadMe is in progress --
## 1. Problem Statement

Language modeling is the task of predicting the next word in a sequence of text, given the previous words. It is a fundamental problem in natural language processing (NLP) with a wide range of applications. 
  Effective language models can be used to power intelligent assistants, improve machine translation, generate realistic text, enhance speech recognition, and even assist with code completion. 
  These models learn patterns in language by training on large text corpora, allowing them to capture the complexities of human communication. 
  Advancements in language modeling have been a driving force behind the recent progress in various NLP tasks, making it a crucial component in developing more natural and intelligent interactions
  between humans and machines.



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

The dataset used for this project is the [WikiText-2](https://github.com/ft-Azad/Language-Modeling/blob/main/wikitext-2-v1.zip) corpus. The text was tokenized using the "basic_english" tokenizer, which resulted in a vocabulary size of 28,782 unique tokens.

The mean length of the sentences is 21.7, however it's important to note that the titles and headers of paragraphs (e.g. "= = Gameplay = =", "= = = Release = = =") are also considered as individual sentences. When excluding these, the true average sentence length is likely closer to 30.

The 15 most frequent tokens in the dataset are:
```
('the', 130768), (',', 102615), ('.', 83397), ('of', 57030), ('<unk>', 54625), ('and', 50735), ('in', 45015), ('to', 39521), ('a', 36523), ('=', 29570), ('was', 21008), ("'", 18484), ('@-@', 16906), ('on', 15140), ('as', 15058)
```
As can be seen, the `'<unk>'` token, representing unknown words, appears 54,625 times, which is the 5th most frequent token in the corpus. This means that the model should be expected to predict the `'<unk>'` token as the next word fairly often, and this will need to be handled appropriately. Additionally, the `'='` token appears very frequently, which is likely due to the inclusion of section headers and titles that are in the form of "= = Heading = =" as mentioned above.

On the other end of the spectrum, the 15 least frequent tokens appear only 3 times each in the entire dataset:
```
('gallinae', 3), ('intergrades', 3), ('northeasterly', 3), ('tuscola', 3), ('roundabouts', 3), ('zoromski', 3), ('forrester', 3), ('kreutzer', 3), ('prefaced', 3), ('philipp', 3), ('chants', 3), ('sonatine', 3), ('mineurs', 3), ('étude', 3), ('caprices', 3)
```
If a minimum frequency threshold of 4 were defined for the vocabulary, the vocabulary size would decrease from 28,782 to 23,652 unique tokens. This means that 5,130 tokens (28,782 - 23,652) are used only 3 times in the entire dataset. With this change that leads to a smaller vocab, the complexity is reduced and training becomes much easier. However, these changes can also lead to a reduction in generalizing ability, and the model may have less creativity. In this project, however, the minimum frequency threshold of 4 is not used, and all 28,782 tokens are included in the vocabulary.

### 4.1. Model
