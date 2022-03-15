# Modeling Intensification for Sign Language Generation
Public repo for the paper: "Modeling Intensification for Sign Language Generation: A Computational Approach" by Mert Inan*, Yang Zhong*, Sabit Hassan*, Lorna Quandt, Malihe Alikhani

## Abstract
End-to-end sign language generation models do not accurately represent the prosody that exists in sign languages. The lack of temporal and spatial variation in the models’ scope leads to poor quality and lower human understanding of generated signs. In this paper, we seek to improve prosody in generated sign languages by modeling intensification in a data-driven manner. We present different strategies grounded in linguistics of sign language that differ in how intensity modifiers can be represented in gloss annotations. To employ our strategies, we first annotate a subset of the benchmark PHOENIX14T, a German Sign Language dataset, with different levels of intensification. We then use a supervised intensity tagger to extend the annotated dataset and obtain labels for the remaining portion of the
dataset. This enhanced dataset is then used to train state-of-the-art transformer models for sign language generation. We find that our efforts in intensification modeling yield better results when evaluated with automated metrics. Human evaluation also indicates a higher preference of the videos generated using our model.

## Organization of the repo
This repo provides the codes and the data required to replicate the results from the paper. The dataset that we provide with the intensifier augmentations to the gloss can be used for further research.

Under the /code folder you can find the codes for the main proposed sign language generation model. In addition, our gloss augmentation codes are also provided, here.

