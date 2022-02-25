# ST456 DL - projects

## Project topics
 
Your project should demonstrate that you have mastered some of the deep learning topics covered in the course, with a focus on methodology, 
neural network architecture design, implementation, training, and evaluation of the performance of a deep learning model. 
Your implementation must be in TensorFlow, using a dataset suitable for your problem.

The project is a group project, typically, a group would consist of three students. You are expected to propose a project topic and form groups among yourself. 
We will provide some means to facilitate group formation and agreeing project topic --- more details to follow. Your project proposal must be approved by the course lecturer. 
You will be informed how to make your project proposal in due course. 

You are expected to split the work on your project among yourself. It is expected from each group member to make a fair share of technical contributions to the project. 

Your report would typically be in the form of a Jupyter notebook, containing TensorFlow code with explanations, along with a Markdown text explaining different 
parts if needed. You may also want to write your report in pdf format, which would give you more flexibility in formatting of your project report --- this will be in addition to your code source files (either standalone or in a Jupyter notebook). 

Your are expected to define and explain concepts used in your project to demonstrate that you understand them.  

You will be assigned a dedicated private GitHub repo for your project. Your report and any other project-related materials should be made available in this GitHub repo.  
 
It is expected from your report to be presented up to a high professional standard. This means that it has to be well structured, neat and polished. 
Your report should have a title, abstract, introduction, methodology, numerical evaluation, and conclusion section, followed by bibliography --- as standard for research papers.  In the abstract, please make sure to clearly describe what is the problem studied in your report, why is this problem a problem (why is it non-trivial), what is your solution, and what are your main results. The abstract should be short, a paragraph of 5-10 sentences. You may use visualizations in your report, for example using Matplotlib, Tensorboard and other Python libraries. Your report must cite any references that you use. You may also discuss and cite any previously-proposed solutions to your problem, and compare the performance of your solution with the performance of other solutions used as baselines in the numerical evaluation section.
The conclusion section should briefly summarise the main results of the report, and briefly discuss interesting directions for future research. At the end of your report, you must add a section with the title "Statement about individual contributions", in which you need to summarise individual technical contributions of each group member. 


In what follows, we provide some examples of project topics and some references, from which you may draw some inspiration. You are not expected to choose a project topic from these suggestions ! We'd like to encourage you to try come up with some original project topic idea (not suggested by what is given below).

## Examples and references

Here you'll some find references to various resources such as research papers and blogs that may inspire your choice of the project topic. You may also check references provided in the lecture and seminar materials.

You may check this page recurrently as more references may be added throughout the course.

The references are presented in no particular order. 

#### Understanding neural networks / interpreting predictions

* Schwartz-Ziv and Tishby, [Opening the black box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810), 2017
* Ribeiro, Sing and Guestrin, [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf), ACM KDD 2016; [github repo](https://github.com/marcotcr/lime)

#### Image classification

* [Patch-based Convolutional Neural Network for Whole Slide Tissue Image
Classification](https://openaccess.thecvf.com/content_cvpr_2016/papers/Hou_Patch-Based_Convolutional_Neural_CVPR_2016_paper.pdf) [MV added 24 Feb 2022]
* PlantVillage [dataset](https://www.tensorflow.org/datasets/catalog/plant_village) [MV added 25 Jan 2022]
* Mohanty, Huges, and Salathe, [Using Deep Learning for Image-Based Plant Disease Detection](https://arxiv.org/pdf/1604.03169.pdf), see also [here](https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full) [MV added 25 Jan 2022] 
* [Aligning Books and Movies](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhu_Aligning_Books_and_ICCV_2015_paper.pdf) [MV added 28 March 2020]
* [FastMRI initiative releases neuroimaging data set](https://ai.facebook.com/blog/fastmri-releases-neuroimaging-data-set/)
* Gurovich et al [Identifying facial phenotypes of genetic disorders using deep learning](https://www.nature.com/articles/s41591-018-0279-0?error=cookies_not_supported&code=810c6851-7b27-4402-b84d-a8fbe2a7819c), Nature Medicine, 2018
* Bien et al, [MRNet: Deep-learning-assisted diagnosis for knee magnetic resonance imaging](https://stanfordmlgroup.github.io/projects/mrnet/), PLOS Medicine, 2018
* ResNet in TensorFlow https://github.com/tensorflow/models/tree/master/official/resnet

#### Natural language processing

* SuperGLUE language benchmark https://venturebeat.com/2021/01/06/ai-models-from-microsoft-and-google-already-surpass-human-performance-on-the-superglue-language-benchmark/?fbclid=IwAR2WX-19ySrys9aULxiPHfCbnTTA3flLAE3FfUkqXPQCop3CKY3DEyeaCYU [MV added 7th January 2021]

#### Set functions

* Zaheer et al, [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html), NIPS 2017 [MV added 25 Jan 2022]
* Wagstaff et al, [On the Limitations of Representing Functions on Sets](http://proceedings.mlr.press/v97/wagstaff19a/wagstaff19a.pdf), ICML 2019 [MV added 25 Jan 2022]
* Segol and Lipman, [On Universal Equivariant Set Networks](https://openreview.net/pdf?id=HkxTwkrKDB), ICLR 2020 [MV addded 25 Jan 2022]

#### Sequence models

* Vinyals, Fortunato, and Jaitly, [Pointer networks](https://papers.nips.cc/paper/5866-pointer-networks.pdf), NIPS 2015, [fastml intro](http://fastml.com/introduction-to-pointer-networks/)
* Vinyals, Bengio, and Kudlur, [Order matters: sequence to sequence for sets](https://arxiv.org/pdf/1511.06391.pdf), ICLR 2016
* van den Oord et al, [Wavenet: a generative model for raw audio](https://arxiv.org/pdf/1609.03499.pdf), arXiv 2016, [blog](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

#### Conversational AI
* Visual question answering, VQA [paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf), [website](https://visualqa.org/) [MV added 29 March 2020]
* [SQuAD 2.0 The Stanford Question Answer Dataset](https://rajpurkar.github.io/SQuAD-explorer/) [MV added 28 March 2020]
* Galley and Gao, Neural Approaches to Conversational AI, ICML 2019 [tutorial](https://icml.cc/Conferences/2019/ScheduleMultitrack?event=4342)
* Conversational AI: Haixun Wang's [blog](https://medium.com/gobeyond-ai/a-reading-list-and-mini-survey-of-conversational-ai-32fceea97180)

#### Graph neural networks, classification, link prediction, alignment
* Bronstein et al, [Geometric Deep Learning Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/pdf/2104.13478.pdf) [MV added 25 Jan 2022]
* Hamilton, [Graph Representation Learning Book](https://www.cs.mcgill.ca/~wlh/grl_book/) [MV added 25 Jan 2022]
* Kipf and Welling, [Semi-supervised classification with graph convolutional networks](https://arxiv.org/pdf/1609.02907.pdf), ICLR 2017
* Defferrard, Bresson, and Vandergheynst, [Convolutional neural networks on graphs with fast localized spectral filtering](https://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf), NIPS 2016
* Monti et al, [Geometric deep learning on graphs and manifolds using mixture model CNNs](http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf), CVPR 2017
* Hamilton, Ying and Leskovec, [Representation learning on graphs: methods and applications](https://www-cs.stanford.edu/people/jure/pubs/graphrepresentation-ieee17.pdf), Proc. of IEEE, 2017
* Leskovec et al, [Tutorial: Deep learning for network biology](http://snap.stanford.edu/deepnetbio-ismb/)
* Knowledge graph entity alignment: [Combining knowledge graphs, quickly and accurately](https://www.amazon.science/blog/combining-knowledge-graphs-quickly-and-accurately) [MV added 20 March 2020]
* [PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph), [paper](https://mlsys.org/Conferences/2019/doc/2019/71.pdf) [MV added 14 Feb 2021]

#### Healthcare

* [A Guide to Deep Learning in Healthcare](https://med.stanford.edu/content/dam/sm/dbds/documents/biostats-workshop/s41591-018-0316-z.pdf) [MV added 22 Feb 2022]
* [Improving the Accuracy of Genomic Analysis with DeepVariant 1.0](https://ai.googleblog.com/2020/09/improving-accuracy-of-genomic-analysis.html#:~:text=DeepVariant%20is%20a%20convolutional%20neural,as%20an%20image%20classification%20problem.) [MV added 17 Feb 2022]
* [Biobank](https://www.ukbiobank.ac.uk/enable-your-research/about-our-data/imaging-data) [MV added 17 Feb 2022]
* https://doctorpenguin.com/ [MV added 17 Feb 2022]
* Rajpukar et al, [AI in Health and Medicine](https://www.nature.com/articles/s41591-021-01614-0), Nature Medicine, 2022 [MV added 16 Feb 2022] 
* Esteva et al, [Deep learning-enabled medical computer vision](https://www.nature.com/articles/s41746-020-00376-2), npj Digital Medicine, 2021 [MV added 27 Jan 2022]
* Rajkomar et al, [Scalable and accurate deep learning with electronic health records](https://www.nature.com/articles/s41746-018-0029-1%22), npj Digital Medicine, 2018 [MV added 26 Jan 2022]
* Landi et al, [Deep representation learning of electronic health records to unlock patient stratification at scale](https://www.nature.com/articles/s41746-020-0301-z), npj Digital Medicine, 2020 [MV added 26 Jan 2022]
* [AI Cures](https://www.aicures.mit.edu/) [MV added 28 March 2020]


#### Optimization

* [TraceIn](https://ai.googleblog.com/2021/02/tracin-simple-method-to-estimate.html#:~:text=TracIn%20is%20a%20simple%2C%20easy,github%20linked%20in%20the%20paper.) A Simple Method to Estimate the Training Data Influence [MV added 16 Feb 2021] 
* Microsoft [MARO (Multi-Agent Resource Optimization)](https://github.com/microsoft/maro) [MV added 3 January 2021]
* Kool et al, [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm) [MV added 20 March 2020]
* Mao et al, [Resource management with deep reinforcement learning](https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf), Hotnets 2016
* Mirhoseini et al, [Device placement optimization with reinforcement learning](https://arxiv.org/abs/1706.04972), ICML 2017
* Mirhoseini et al, [A hierarhical model for device placement](https://openreview.net/pdf?id=Hkc-TeZ0W), ICLR 2018
* Bello et al, [Neural combinatorial optimization with reinforcement learning](https://arxiv.org/pdf/1611.09940.pdf), ICLR 2017

#### Protein structure prediction

* Senior et al, [AlphaFold: Improved protein structure prediction using potentials from deep learning](https://deepmind.com/research/publications/AlphaFold-Improved-protein-structure-prediction-using-potentials-from-deep-learning)

#### Finance

* Deng et al, [Deep direct reinforcement learning for financial
signal representation and trading](http://www.cslt.org/mediawiki/images/a/aa/07407387.pdf), IEEE Trans. on Neural Networks and Learning Systems, 2016
* Heaton, Polson and Witte, [Deep learning in finance](https://arxiv.org/pdf/1602.06561.pdf), ArXiv, 2016


#### Generative models

* [Nowcasting: making short-term weather predictions](https://deepmind.com/blog/article/nowcasting) [MV added 25 Jan 2022]

#### Self-supervised learning

* Meta AI, [The first high-performance self-supervised algorithm that works for speech, vision, and text](https://ai.facebook.com/blog/the-first-high-performance-self-supervised-algorithm-that-works-for-speech-vision-and-text/) [MV added 26 Jan 2022]


#### Recommendation systems

* Naumov et al, [Deep Learning Recommendation Model for
Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091.pdf), GitHub: [dlrm](https://github.com/facebookresearch/dlrm) [MV added 25 Jan 2022]

### Some past project topics

**Note:** prior to 2021/22, the course covered the topic of reinforcement learning, and did not cover autoencoders and generative models

#### 2020/21

* Abstractive Text Summarization with Transfer Learning
* An application of the novel DeepCascade-WR for forecasting volatile time series
* Applying Reinforcement Learning to explore optimal solutions for the container loading problem (CLP)
* Artificial Intelligence – a Threat to Cryptocurrency Investment Advisors?
* Classify similar products into groups using product images and titles
* Combining sentiment analysis of web articles and past financial data for stock price prediction using deep learning
* Comparsion between ReLu and Sigmod: how activation functions perform under different depth and neuron units
* Conversational model based on seq2seq framework
* Colorizing Greyscale Images with Neural Networks
* COVID-19 detection using deep learning techniques
* Deep Learning for Natural Language Processing: Large Movie Review Sentiment Analysis using various Deep Neural Networks in Keras
* Deep reinforcement learning for algorithmic trading
* Detecting Cyberbullying using Neural Networks for Text Classification
* Forcasting Boston House Prices using XGBoost and LSTM
* Gender Classification Based on First Names
* Image Caption Recommender System
* Image classification for diagnosing Covid-19 & Pneumonia by Chest X-ray
* Learning Tic-tac-toe
* LSTM with Emotional Analysis for Stock Price Prediction
* Option pricing using deep learning
* Pneumonia Classification
* Portfolio Trading with Deep Reinforcement Learning
* Prediction of sharing disinformation based on the neuron network
* Quantitative trading by reinforcement learning
* Sentiment Analysis for Short-Term Stock Price Predictions
* Sentiment classification of Covid-19 related posts on Twitter
* Vehicle Detection and Classification using Yolov4 and SSD
* Yolov3 adaptation for Object Segmentation


#### 2019/20

* Abstract text summarization using RNNs
* Anomaly detection in images with object localization
* BERT: interpretatability and explainability
* Conversational ChatBot using deep learning
* Classification and depression-indicating language in written text
* Classification of plant dieseases
* Creating a conversational ChatBot using deep Q-network
* Detecting sentiment using LSTMs and CNNs
* Detection of knee injuries using deep learning techniques
* Diabetic retinopathy detection
* Exploring the ResNet architecture
* Fairness or efficiency: strategy analysis for coronavirus medical treatment using RL
* Financial portfolio management using deep RL
* Generating synthetic data using GANs
* Heart anomaly detection
* Image classification for diagnosing pneumonia by Chest X-ray images
* Multi-class classification of news category with BERT
* Music genre recognition using neural networks
* Predicting affective content in tweets with deep attentive RNNs
* Prediction of meterological data using LSTMs and LSTM-CNNs
* Recurrent neural networks and transformers in bilingual machine translation
* Sentiment analysis on Amazon fine food reviews dataset
* Sentiment analysis of Amazon reviews
* Smart indexing using autoencoders
* Stock prediction with recurrent neural networks
* The use of deep neural networks to predict NBA game outcomes
* Using behavioral patterns in recommender systems
* Using knowledge distillation to increase accuracy of lightweight CNNs for image classification
* Variable selection using deep neural networks

#### 2018/19

* Author identification with bidirectional recurrent neural networks
* Capsule networks in deep learning
* Classification of pigmented skin lesions
* Deep direct recurrent reinforcement learning for algorithmic trading
* Densely connected convolutional neural network for mammography and invasive ductal carcinoma histology
* Exploring image classification techniques to predict poverty levels
* Learning Boolean functions
* Music generation with artificial intelligence-creative sequence modelling using LSTM and recurrent neural networks
* Measuring political preference with sparse text classification methods
* Negotiation agents
* Neural machine translation with attention weights
* ResNet architecture for image classification
* Predicting hurricane trajectories using RNN
* Predicting systemic financial crises with recurrent neural networks
* Reinforcement learning for trade execution with Alpha and risk aversion
* Sentiment classification of large movie review dataset
* Sentiment analysis on Amazon book reviews dataset using RNNs and LSTM
* Seq2Slate model: ranking with RNNs and sequential decoding
* Solving ATT48 by deep reinforcement learning
* Stock trading by deep reinforcement learning
