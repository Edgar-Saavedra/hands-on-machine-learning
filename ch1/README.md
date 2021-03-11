[source-code](https://github.com/ageron/handson-ml2)

### Important Packages

**Scikit-learn**
Implements many famous algorithms

**Tensor Flow**
made by google, to run large neural networks

**keras**
high level deep leearning API; makes it simple to train and run neural networks; can run on top of either `TensorFlow`, `Theano`, `Microsoft Cognitive Toolkit`


### Types of machine learning Systems

- **Supervised**
  Trained with human supervision. You feed the algorithm the training set and desired solutions (labels). A classification task.
  - Vocab:
    - labels : desired solutions
    - training set: data used to train ml
    - attribute: a data type ("milage", "age" etc)
    - features: attributes of data. An attribute plus its value
    - regression task: to perdict a value
    - target: you regression target
  - Algorithms/Models
    - k-Nearest Neighbors
    - Liear Regression
    - Logistic Regression
    - Support Verctor Machines (SVMs)
    - Decision Trees and Random Forests
    - Neural Networks
- **unsupervised**
  The training data is unlabeled, the system tries to learn with out a teacher
  - Vocab:
    - dimensionality reduction: merge features
    - feature extraction: dimensionality reduction
  - Notes: 
    - goode idea to reduce dimension of training data using dimensionality reduction
  - Examples:
    - Detecting similar groups
    - Visualization algorithms
  - Algorithms/Models
    - Clustering
      - K-Means
      - DBSCAN
      - Hierarchical Cluster Analysis (HCA)
    - Anomaly detection and novelty detection
      - One-class SVM
      - Isolation Forest
    - Visualization and dimensionality reduction
      - Principal Component Analysis (PCA)
      - Kernel PCA
      - Locally Linear Embedding (LLE)
      - t-Distribution Stochastic Neighbor Embedding (t-SNE)
    - Association rule learning
      - Apriori
      - Eclat
- **online**
- **versus batch learning**
- **instance based**
- **model based learning**

### ML Example tasks
- **Analyze images of products on a production line and classify them**
  Convolutional Neural Networks (CNNs)
- **Semantic segmentation**
  Each pixel in the image is classified (CNNs)
- **Natural Language Processing**
  Recurrent neural networks (RNNs), CNNs or Tranformers
- **Chatbot and personal assistan**
  Natural Language Understanding
- **Forcasting on Metrics**
  Regression (prediction) task. Linear Regression Model, Polynomial Regression, Artificial Neural Networks. Take into account sequences with RNNs, CNNs or Transformers
- **Speach Recognition**
  RNNs & CNNs
- **Recommender systems**
  Artificial Neural Networks
- **Intelligent bot for a game**
  Reinforcement Learning  

### Data

**Training set**
The examples that the system uses to learn

### Work flow
Study problem; Write Rules; Evaluate; Analyze Errors ... Launch

