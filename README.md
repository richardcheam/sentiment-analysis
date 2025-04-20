# Project Overview

In this project, the primary focus is to explore and experiment with language models, particularly Mixture of Experts (MoE), and use these models for feature extraction and embedding. 
Language models like DistilBERT, DeBERTa, RoBERTa are used to understand and process text in a variety of languages, and MoE plays a key role in improving the model's performance by selectively using different "experts" for different tasks.

Additionally, I explored more traditional approaches, such as using TF-IDF (Term Frequency-Inverse Document Frequency) embeddings and feeding them into an LSTM (Long Short-Term Memory) network for sentiment classification. 
These experiments help understand both the power of pretrained models and the effectiveness of simpler, classical methods.

## What is MoE (Mixture of Experts)?

MoE (Mixture of Experts) is a technique that allows a model to use different specialized "experts" (smaller sub-models) for different parts of the data. Imagine if instead of having one big brain trying to do everything, 
you could have a team of experts, each of whom is really good at one specific task. The MoE system picks the most relevant experts based on the input data, leading to more efficient and faster processing.

In the case of language models, MoE allows the model to learn and apply specialized knowledge to various aspects of language, like syntax, semantics, sentiment, or context. This means the model doesn't need to rely on one general model 
for all types of language tasks but can dynamically choose the most suitable expert.
