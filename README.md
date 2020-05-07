# App_Offer_Optimization_Starbucks

Optimizing the usage of offers on the Starbucks Reward App using offer portifolio, customer and transactions data.

Udacity - Data Scientist Nanodegree Capstone Project

>Starbucks Corporation is an American multinational chain of coffeehouses and roastery reserves headquartered in Seattle, Washington.

>As the largest coffeehouse in the world, Starbucks is seen to be the main representation of the United States' second wave of coffee culture.

>Since the 2000s, third wave coffee makers have targeted quality-minded coffee drinkers with hand-made coffee based on lighter roasts, while Starbucks nowadays uses automatic espresso machines for efficiency.

>The company operates 30,000 locations worldwide in over 77 countries, as of early 2020. Starbucks locations serve hot and cold drinks, whole-bean coffee, microground instant coffee known as VIA, espresso, caffe latte, full- and loose-leaf teas including Teavana tea products, Evolution Fresh juices, Frappuccino beverages, La Boulange pastries, and snacks including items such as chips and crackers; some offerings (including their annual fall launch of the Pumpkin Spice Latte) are seasonal or specific to the locality of the store.

**Starbucks Article on Wikipedia**

One convenient way to pay in store, order ahead for pickpup or even get updated about new drinks or items, is the Starbucks Rewards App. Rewards are built right in, so you’ll collect Stars and start earning free drinks and food with every purchase.

The data sets used in this project contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

### Goal:

Not all users receive the same offer, and that is the challenge this project aims to solve:

The task here is to combine **transaction**, **demographic** and **offer data** to determine **which demographic groups respond best to which offer type**. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

### More details:

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. One'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

One'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer.

It's worth to keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.

## Project Steps

The project were build according to the following structure:

* Business Understanding : Problem definition;
* Data Understanding : Data dictionary for all the three datasets used;
* Exploratory Data Analysis : ETL Pipeline to prepare data from all the three datasets to be used into the Customer classification step;
* Customer Classification : Label customers according to the effectiveness of each type of offer;
* Data Pre Processing : Prepare the labeled data to fit the classification model;
* Modeling : Multi-Class Multi-Label Machine Learning algorithm to predict customer preferences regarding app offers;
* Evaluation : Model evaluation according to the chosen metrics;
* Conclusions : Did the proposed method work? Is it worth to deploy it? How to improve?

## Installation

The project were done entirely using a Jupter Notebook on through Google Colab, hence there's no need of installation steps.

## Data

Dataframes used are available on the links below:

*   portifolio_df: https://drive.google.com/open?id=1J2NRI-js0MhcnMEkdT9yLScmoA6GdIVl
*   profile_df: https://drive.google.com/open?id=19FWNvSjVeFMExM3vqokdI0ZHkcWI4LcB
*   transcript_df: https://drive.google.com/open?id=1N8NsO1UDMDYjhTX2J-UTLH5c32E-CHRk

## Libraries

The following libraries were used during the project:

**Data**

*   PanDas
*   json

**Utility**

*   Matplotlib
*   Seaborn
*   Numpy

**Unsupervised Machine Learning**

*   Sklearn.mixture
*   mpl_toolkits.mplot3D

**Supervised Machine Learning**

*   Sklearn (preprocessing, model_selection, Pipeline, svm, multiclass, multioutput, model_selection, metrics)

## Results

The best results were achieved using the following machine learning pipeline

```
pipeline = Pipeline([
        ('clf',MultiOutputClassifier(OneVsRestClassifier(LinearSVC(), n_jobs=1)))  
                        ])
# Define hyperparameters to be optimized in the GridSearchCV
parameters = {
    'clf__estimator__estimator__loss': ('squared_hinge', 'hinge'),
    'clf__estimator__estimator__multi_class': ('ovr','crammer_singer'),
    'clf__estimator__estimator__max_iter': (500,1000,2000,5000)
             }

# Define scoring metrics for the GridSearchCV optimization
scoring = {
    'accuracy' : make_scorer(accuracy_score),
    'average_precision' : make_scorer(average_precision_score)
        } 

# Optimize and Fit
cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, cv=5, scoring=scoring, refit='average_precision')
cv.fit(X_train, y_train)

```

GridSearch CV has found the following hyperparameters as the ones which optimized the results according to the specified criteria:

```
C=1.0,
class_weight=None,
dual=True,
fit_intercept=True,
intercept_scaling=1,
loss='squared_hinge',
max_iter=1000,
multi_class='ovr',
penalty='l2',
random_state=None,
tol=0.0001
```

Metrics per category:

| Category | Precision | Recall | F1-Score | Support |
| ------ | ------ | ------ | ------ | ------ |
| related | 0.57 | 1.00 | 0.72 | 752 |
| request | 0.60 | 1.00 | 0.75 | 788 |
| offer | 0.57 | 1.00 | 0.72 | 752 |
|  |  |  |  |  |
| micro avg | 0.58 | 1.00 | 0.73 | 2292 |
| macro avg | 0.58 | 1.00 | 0.73 | 2292 |
| weighted avg | 0.58 | 1.00 | 0.73 | 2292 |
| samples avg | 0.58 | 1.00 | 0.71 | 2292 |

Results achieved at this version were not satisfatory enough for deployment.

**Version 2.0 of the model is going to try different equations for offer metrics, adding some penalty factors for non responded and non completed offers**

## Acknowledgements

* [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
*   [Oscar Contreras Carrasco](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95)
