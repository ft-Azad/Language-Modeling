

<h1 align="center">Language Modeling</h1>
</div>


## 1. Problem Statement

Language modeling is the task of predicting the next word in a sequence of text, given the previous words. 
It's a fundamental problem in natural language processing (NLP) with a wide range of applications. Effective language
 models can power intelligent assistants, improve machine translation, generate realistic text, enhance speech recognition,
 and even assist with code completion. These models learn patterns in language by training on large text corpora, allowing
 them to capture the complexities of human communication. Advancements in language modeling have been a driving force 
behind recent progress in various NLP tasks, making it a crucial component in developing more natural and intelligent 
human-machine interactions.

In this project, a language model is trained to take a prompt as input from the user and complete the text. The model 
uses deep learning to learn from a large corpus of text data, enabling it to generate coherent and contextually 
appropriate text continuations.



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
If a minimum frequency threshold of 4 were defined for the vocabulary, the vocabulary size would decrease from 28,782 to 23,652 unique tokens. This means that 5,130 tokens (28,782 - 23,652) are used only 3 times in the entire dataset. With this change that leads to a smaller vocab, the complexity is reduced and training becomes much easier. However, these changes can also lead to a reduction in generalizing ability, and the model may have less creativity. 

In this project, the minimum frequency threshold of 4 is not used, and all 28,782 tokens are included in the vocabulary. For making the train and label for the model, the raw text `seq_len` tokens are considered as input, and the next token is used as the output. For the base model, the sequence length used is 35 tokens.

### 4.2. Model
For the base model, the first layer is an embedding layer with a size of 300. After the embedding layer, a dropout layer with a rate of 0.5 is added. Then, a 2-layer LSTM is used. For the LSTM layers, an rnn_dropout of 0.2 is used. Finally, a fully connected layer is applied to the output of the LSTM, with an output size equal to the vocabulary length.

The summary of the model is:
```
LanguageModel(
  (embedding): Embedding(28782, 300)
  (dropout): Dropout(p=0.5, inplace=False)
  (lstm): LSTM(300, 512, num_layers=2, batch_first=True, dropout=0.2)
  (fc): Linear(in_features=512, out_features=28782, bias=True)
)
```
For the improved model, the following changes were made to the base architecture:

1. The embedding layer size was increased from 300 to 400 dimensions.
2. Instead of a traditional dropout layer, a Locked Dropout layer with a rate of 0.65 was used after the embedding layer.
3. The LSTM was expanded to 3 layers, with an input/output size of 400 and hidden layers size of 1150.
4. After each LSTM layer, a Locked Dropout layer was added with a rate of 0.3. For the final LSTM layer, the Locked Dropout rate was increased to 0.4.
5. The fully connected layer at the end has the same weight as the embedding layer, a technique called weight tying, where the weights are shared and trained simultaneously.
6. Additionally, a weight dropout of 0.5 was added to the recurrent weights of the LSTM layers.

The overall model architecture can be summarized as:
```
LanguageModel(
  (embedding): Embedding(23652, 400)
  (embedding_dropout): LockedDropout(p=0.65)
  (lstms): ModuleList(
    (0): WeightDrop(
      (module): LSTM(400, 1150)
    )
    (lstm0_dropout): LockedDropout(p=0.3)
    (1): WeightDrop(
      (module): LSTM(1150, 1150)
    )
    (lstm1_dropout): LockedDropout(p=0.3)
    (2): WeightDrop(
      (module): LSTM(1150, 400)
    )
    (lstm2_dropout): LockedDropout(p=0.4)
  )
  (fc): Linear(in_features=1150, out_features=23652, bias=True)
)
```
### 4.3. Configurations
For the training process, cross-entropy was used as the loss function and perplexity was used as the model performance metric. The SGD optimizer was employed for training.

The base model had a learning rate of 3, a weight decay of 1e-6, and a momentum of 0.9. The batch size was 20 and the sequence length was 35.

The improved model utilized a higher learning rate of 7.5, again with a weight decay of 1e-6 and a momentum of 0.9. The batch size was increased to 80 and the sequence length was increased to 70. The higher learning rate for the improved model was enabled by the additional regularization techniques that were introduced.

To ensure stability during training, gradient clipping was applied with a clip value of 0.25 during the backward pass.

### 4.4. Train
The learning curves clearly demonstrate the impact of the additional regularization techniques introduced in the improved model. For access to the learning curves, click [here](https://www.comet.com/ft-azad/lm-awd-lstm/view/new/panels).

Without the regularization, the base model's training loss dropped significantly early on, indicating a tendency to overfit the training dataset. This is a common issue that can negatively impact the model's generalization performance on unseen data.

In contrast, the improved model with the added regularization showed a much more gradual descent in the training loss. The loss did not drop too rapidly, allowing the model to generalize better and avoid premature overfitting.

This difference in the learning curves illustrates how the regularization techniques were crucial in enabling the improved model to learn more robust and generalized representations, rather than simply memorizing the training data.

![Learning Curve](https://github.com/ft-Azad/Language-Modeling/blob/main/results/metric_train%20VS%20epoch.jpeg)

### 4.4. Evaluate

As can be seen in the validation curves, which are available [here](https://www.comet.com/ft-azad/lm-awd-lstm/view/new/panels), changing the sequence length and batch size had a significant positive impact on the model's performance metrics. Increasing the sequence length allowed the model to better capture contextual information, leading to improved perplexity scores. Similarly, the larger batch size facilitated more generalized learning steps, further enhancing the model's capabilities.

However, when introducing a new architectural change that included increasing the embedding size, adding an extra LSTM layer, and using higher hidden dimensions for the LSTMs, the performance metrics actually decreased and became less stable. This was likely due to the larger number of learning parameters. Most of these parameters were caused by the embedding layer and the final fully connected layer. The size of these layers was defined by the vocabulary length and the embedding length, which led to a very large number of parameters. The model struggled to learn these parameters effectively with the given dataset size.

After implementing the weight tying technique, the parameters for the embedding layer and the final fully connected layer became one and the same. This meant the model only needed to learn half the number of parameters for encoding and decoding the tokens. This change had a dramatic positive impact, allowing the model to achieve the best perplexity scores by a wide margin compared to the previous configurations.

To prevent overfitting as training progressed, techniques such as embedding dropout, locked dropout, and weight drop were introduced. While these regularizers did cause the model to learn more slowly initially, they ultimately allowed the model to continue improving even as the unregularized model started to overfit. The regularized model was able to achieve much better perplexity scores over longer training times and higher epochs compared to the previous configurations.

![validation Curve](https://github.com/ft-Azad/Language-Modeling/blob/main/results/metric_valid%20VS%20epoch.jpeg)

### 4.5. Test and Generate
The loss obtained on the test set is 86.92, which is not the best that can be achieved, but the effect of the applied techniques can be clearly seen compared to the base model.

For generating text and sampling the next token, the following approach was used:

1. First, a softmax with a temperature of 0.5 was applied to the model's output.
2. Then, a token was chosen using a multinomial distribution.
   
This approach introduced a bit of randomness in predicting the next word, which made the model more creative and able to generate more varied text with a similar prompt.

An example generation from this model is shown below:

"Before going to bed, she"

``` Before going to bed, she was able to make a call to the group . ```
