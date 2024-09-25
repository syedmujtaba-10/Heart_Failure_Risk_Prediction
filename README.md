![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)


# Using Machine Learning to Predict Survival of Patients with Heart Failure

## Table of contents
   * [Overview](#Overview)
   * [System Requirements](#System-Requirements)
   * [Project Set Up and Installation](#Project-Set-Up-and-Installation)
   * [Dataset](#Dataset)
   * [Automated ML](#Automated-ML)
   * [Hyperparameter Tuning](#Hyperparameter-Tuning)
   * [Model Deployment](#Model-Deployment)
   * [Flask Deployment](#Flask-Deployment)
   * [Chatbot integration](#Chatbot-Integration)
   * [Screen Recording](#Screen-Recording)
   * [Dataset Citation](#Dataset-Citation)
   * [References](#References)

***

## Overview

We created two models in the environment of Azure Machine Learning Studio: one using Automated Machine Learning (i.e. AutoML) and one customized model whose hyperparameters are tuned using HyperDrive. We then compare the performance of both models and deploy the best performing model as a service using Azure Container Instances (ACI).

The diagram below is a visualization of the rough overview of the operations that take place in this project:

![Project Workflow](images/Project_workflow.JPG?raw=true "Project Workflow") 


## System Requirements

● Microsoft Azure:
  Microsoft Azure is a cloud computing platform that provides a wide range of services for
  building, deploying, and managing applications and services through Microsoft-managed
  data centers. Azure offers a suite of cloud services, including compute, storage, networking,
  databases, analytics, and machine learning.
  Azure Machine Learning is a cloud-based service that enables developers and data scientists
  to build, deploy, and manage machine learning models at scale. With Azure Machine
  Learning, users can use a range of tools and frameworks such as Python, R, and TensorFlow
  to create and deploy machine learning models.
  
  Azure Machine Learning provides functionality for the following tasks:
  
    1. Data preparation: Azure Machine Learning provides tools for cleaning, transforming,
      and pre-processing data for machine learning models.
      
    2. Model training and tuning: Azure Machine Learning provides tools for training
      machine learning models and tuning their hyperparameters.
      
    3. Model deployment: Azure Machine Learning provides tools for deploying machine
      learning models to a variety of environments, including cloud-based web services,
      containers, and edge devices.
      
    4. Model monitoring and management: Azure Machine Learning provides tools for
      monitoring and managing machine learning models in production, including
      monitoring model performance and retraining models when necessary.


● Flask:
  Flask is a popular web framework for building web applications and APIs in Python. Flask
  provides a simple and flexible way to handle HTTP requests and responses, and its
  lightweight design makes it easy to deploy and scale.
  
  When deploying a Flask application, there are several options available. Here are some
  common deployment options:
  
    1. Deploying on a traditional web server: Flask applications can be deployed on
      traditional web servers like Apache or Nginx. This typically involves using a WSGI
      server, such as Gunicorn, to handle the Flask application and proxy requests to the
      web server.
      
    2. Deploying on a Platform as a Service (PaaS) provider: PaaS providers, such as
      Heroku or Google App Engine, offer managed environments for deploying web
      applications. These platforms provide preconfigured runtimes and services, making it
      easy to deploy and scale Flask applications without needing to manage infrastructure.
      
    3. Deploying on a container platform: Flask applications can also be deployed in
      containers, such as Docker containers, and managed with container orchestration tools
      like Kubernetes. This approach provides a high level of flexibility and scalability, but
      also requires more advanced knowledge of containerization and orchestration.

● Spyder:
  Spyder is a popular Integrated Development Environment (IDE) for scientific computing in
  Python. It provides a comprehensive development environment for data scientists and
  scientific programmers, with a range of features for editing, debugging, and executing Python
  code.
  
  Some of the key features of Spyder include:
  
    1. Interactive console: Spyder provides an interactive console that allows users to
      execute Python code and view the results in real-time. This is particularly useful for
      exploring data and testing code snippets.
      
    2. Code editor: Spyder includes a powerful code editor that provides features like syntax
      highlighting, code completion, and code linting. It also includes support for multiple
      file formats, including Python, Markdown, and HTML.
      
    3. Debugger: Spyder has a built-in debugger that allows users to step through code and
      inspect variables and data structures at runtime. This makes it easier to find and fix
      bugs in your code.
      
    4. Data exploration tools: Spyder includes tools for data exploration and analysis, such
      as a variable explorer that allows users to view and manipulate data structures, and a
      plotting library for creating visualizations.
      
    5. Integration with scientific libraries: Spyder integrates with popular scientific libraries,
      such as NumPy, SciPy, and Matplotlib, making it a powerful tool for scientific
      computing and data analysis.


● Sublime Text:
  Sublime Text is a popular text editor used by programmers and developers. It is known for its
  sleek and minimalist user interface and powerful features, such as syntax highlighting,
  auto-completion, multiple selections, and the ability to customize shortcuts and preferences.
  Sublime Text supports a wide range of programming languages and file formats, making it a
  versatile tool for various projects. It also has a large community of users who have developed
  plugins and packages to extend its functionality even further. Sublime Text is available for
  Windows, Mac, and Linux, and can be downloaded and used for free, although a license can
  be purchased for additional features and support.


## Project Set Up and Installation

In order to run the project in Azure Machine Learning Studio, we will need the two Jupyter Notebooks:

- `automl.ipynb`: for the AutoML experiment;
- `hyperparameter_tuning.ipynb`: for the HyperDrive experiment.

The following files are also necessary:

- `heart_failure_clinical_records_dataset.csv`: the dataset file. It can also be taken directly from Kaggle; 
- `train.py`: a basic script for manipulating the data used in the HyperDrive experiment;
- `scoring.py`: the script used to deploy the model which is downloaded from within Azure Machine Learning Studio; &
- `env.yml`: the environment file which is also downloaded from within Azure Machine Learning Studio.


## Dataset

The dataset used is taken from [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) and -as we can read in the original [Research article](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)- the data comes from 299 patients with heart failure collected at the Faisalabad Institute of Cardiology and at the Allied Hospital in Faisalabad (Punjab, Pakistan), during April–December 2015. The patients consisted of 105 women and 194 men, and their ages range between 40 and 95 years old.

The dataset contains 13 features:

| Feature | Explanation | Measurement |
| :---: | :---: | :---: |
| *age* | Age of patient | Years (40-95) |
| *anaemia* | Decrease of red blood cells or hemoglobin | Boolean (0=No, 1=Yes) |
| *creatinine-phosphokinase* | Level of the CPK enzyme in the blood | mcg/L |
| *diabetes* | Whether the patient has diabetes or not | Boolean (0=No, 1=Yes) |
| *ejection_fraction* | Percentage of blood leaving the heart at each contraction | Percentage |
| *high_blood_pressure* | Whether the patient has hypertension or not | Boolean (0=No, 1=Yes) |
| *platelets* | Platelets in the blood | kiloplatelets/mL	|
| *serum_creatinine* | Level of creatinine in the blood | mg/dL |
| *serum_sodium* | Level of sodium in the blood | mEq/L |
| *sex* | Female (F) or Male (M) | Binary (0=F, 1=M) |
| *smoking* | Whether the patient smokes or not | Boolean (0=No, 1=Yes) |
| *time* | Follow-up period | Days |
| *DEATH_EVENT* | Whether the patient died during the follow-up period | Boolean (0=No, 1=Yes) |


### Access

First, we made the data publicly accessible in the current GitHub repository via this link:
[https://github.com/HumdaanSyed/Heart-Failure-Risk-Prediction/blob/main/heart_failure_clinical_records_dataset.csv](https://github.com/HumdaanSyed/Heart-Failure-Risk-Prediction/blob/main/heart_failure_clinical_records_dataset.csv)

and then link the dataset: 

![Dataset creation](images/01.png?raw=true "heart-failure-prediction dataset creation")

As it is depicted below, the dataset is registered in Azure Machine Learning Studio:

***Registered datasets:*** _Dataset heart-failure-prediction registered_
![Registered datasets](images/02.png?raw=true "heart-failure-prediction dataset registered")

We are also accessing the data directly via:

```
data = pd.read_csv('./heart_failure_clinical_records_dataset.csv')
```
## Automated ML

***AutoML settings and configuration:***

Automated Machine Learning (AutoML) is the process of automating the end-to-end process of
applying machine learning to real-world problems. AutoML tools enable data scientists and
non-experts to build machine learning models with minimal manual effort, by automating tasks such
as feature engineering, model selection, and hyperparameter tuning.
Some of the key uses of AutoML include:
  1. Faster model development: AutoML tools can significantly reduce the time and effort
    required to develop machine learning models, by automating tasks that would otherwise
    require manual effort.
  2. Increased accessibility: AutoML tools can democratize machine learning by making it more
    accessible to non-experts, who may not have the technical expertise required to develop
    machine learning models from scratch.
  3. Improved model performance: AutoML tools can help to improve the performance of
    machine learning models, by automating tasks such as hyperparameter tuning, which can be
    time-consuming and error-prone when done manually.
  4. Increased scalability: AutoML tools can help to scale machine learning models to larger
    datasets and more complex problems, by automating tasks such as feature engineering, which
    can become increasingly difficult as the complexity of the problem grows.


![AutoML settings & configuration](images/17.JPG?raw=true "AutoML settings & configuration")

Below you can see an overview of the `automl` settings and configuration we used for the AutoML run:

`"n_cross_validations": 2`

This parameter sets how many cross validations to perform, based on the same number of folds (number of subsets). As one cross-validation could result in overfit, in my code I chose 2 folds for cross-validation; thus the metrics are calculated with the average of the 2 validation metrics.

`"primary_metric": 'accuracy'`

I chose accuracy as the primary metric as it is the default metric used for classification tasks.

`"enable_early_stopping": True`

It defines to enable early termination if the score is not improving in the short term. In this experiment, it could also be omitted because the _experiment_timeout_minutes_ is already defined below.

`"max_concurrent_iterations": 4`

It represents the maximum number of iterations that would be executed in parallel.

`"experiment_timeout_minutes": 20`

This is an exit criterion and is used to define how long, in minutes, the experiment should continue to run. To help avoid experiment time out failures, I used the value of 20 minutes.

`"verbosity": logging.INFO`

The verbosity level for writing to the log file.

`compute_target = compute_target`

The Azure Machine Learning compute target to run the Automated Machine Learning experiment on.

`task = 'classification'`

This defines the experiment type which in this case is classification. Other options are _regression_ and _forecasting_.

`training_data = dataset`

The training data to be used within the experiment. It should contain both training features and a label column - see next parameter.

`label_column_name = 'DEATH_EVENT'` 

The name of the label column i.e. the target column based on which the prediction is done.

`path = project_folder`

The full path to the Azure Machine Learning project folder.

`featurization = 'auto'`

This parameter defines whether featurization step should be done automatically as in this case (_auto_) or not (_off_).

`debug_log = 'automl_errors.log`

The log file to write debug information to.

`enable_onnx_compatible_models = False`


### Results

#### Completion of the AutoML run (RunDetails widget): 

![Completion of the AutoML run (Run Details)](images/33.png?raw=true "AutoML completed: RunDetails widget")

![AutoML completed](images/05.png?raw=true "AutoML completed: RunDetails widget")

#### Best model

After the completion, we can see and take the metrics and details of the best run:

![Best run metrics and details](images/30.png?raw=true "Best run metrics and details")

![Best run properties](images/31.png?raw=true "Best run properties")

Best model results:

| AutoML Model | |
| :---: | :---: |
| id | AutoML_19c0a6e2-24a1-4698-b53b-122c354d1c1b |
| Accuracy | 0.8629082774049217 |
| AUC_weighted | 0.9006630037732388 |
| Algorithm | VotingEnsemble |


## Hyperparameter Tuning

For this experiment we used a custom Scikit-learn Logistic Regression model, whose hyperparameters I am optimising using HyperDrive. Logistic regression is best suited for binary classification models like this one and this is the main reason we chose it.

We specify the parameter sampler using the parameters C and max_iter and chose discrete values with choice for both parameters.

**Parameter sampler**

We specified the parameter sampler as such:

```
ps = RandomParameterSampling(
    {
        '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
        '--max_iter': choice(50,100,200,300)
    }
)
```

I chose discrete values with _choice_ for both parameters, _C_ and _max_iter_.

_C_ is the Regularization while _max_iter_ is the maximum number of iterations.

_RandomParameterSampling_ is one of the choices available for the sampler and I chose it because it is the faster and supports early termination of low-performance runs. If budget is not an issue, we could use _GridParameterSampling_ to exhaustively search over the search space or _BayesianParameterSampling_ to explore the hyperparameter space. 

**Early stopping policy**

An early stopping policy is used to automatically terminate poorly performing runs thus improving computational efficiency. I chose the _BanditPolicy_ which I specified as follows:
```
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```
_evaluation_interval_: This is optional and represents the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

_slack_factor_: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. This means that with this policy, the best performing runs will execute until they finish and this is the reason I chose it.


### Results
#### Completion of the HyperDrive run (RunDetails widget):

![HyperDrive RunDetails widget](images/34.png?raw=true "HyperDrive RunDetails widget")


| HyperDrive Model | |
| :---: | :---: |
| id | HD_cbb54a5f-5c10-4231-9ade-a6993093b96b_11 |
| Accuracy | 0.8333333333333334 |
| --C | 0.01 |
| --max_iter | 100 |


## Model Deployment

The deployment is done following the steps below:

* Selection of an already registered model
* Preparation of an inference configuration
* Preparation of an entry script
* Choosing a compute target
* Deployment of the model
* Testing the resulting web service

### Registered model

Using as basis the `accuracy` metric, we can state that the best AutoML model is superior to the best model that resulted from the HyperDrive run. For this reason, I choose to deploy the best model from AutoML run (`best_run_automl.pkl`, Version 2). 

_Registered models in Azure Machine Learning Studio_

![Registered models](images/35.jpg?raw=true "Registered models")

_Runs of the experiment_

![Runs of the experiment](images/36.jpg?raw=true "Runs of the experiment")

### Inference configuration

The inference configuration defines the environment used to run the deployed model. The inference configuration includes two entities, which are used to run the model when it's deployed:

![Inference configuration](images/37.jpg?raw=true "Inference configuration")

- An entry script, named `scoring_file_v_1_0_0.py`.
- An Azure Machine Learning environment, named `env.yml` in this case. The environment defines the software dependencies needed to run the model and entry script.

![Inference configuration](images/38.jpg?raw=true "Inference configuration")

### Entry script

The entry script is the `scoring_file_v_1_0_0.py` file. The entry script loads the model when the deployed service starts and it is also responsible for receiving data, passing it to the model, and then returning a response.
### Compute target

As compute target, I chose the Azure Container Instances (ACI) service, which is used for low-scale CPU-based workloads that require less than 48 GB of RAM.

The AciWebservice Class represents a machine learning model deployed as a web service endpoint on Azure Container Instances. The deployed service is created from the model, script, and associated files, as I explain above. The resulting web service is a load-balanced, HTTP endpoint with a REST API. We can send data to this API and receive the prediction returned by the model.

![Compute target](images/39.jpg?raw=true "Compute target")

`cpu_cores` : It is the number of CPU cores to allocate for this Webservice. Can also be a decimal.

`memory_gb` : The amount of memory (in GB) to allocate for this Webservice. Can be a decimal as well.

`auth_enabled` : I set it to _True_ in order to enable auth for the Webservice.

`enable_app_insights` : I set it to _True_ in order to enable AppInsights for this Webservice.
### Deployment

Bringing all of the above together, here is the actual deployment in action:

![Model deployment](images/06.png?raw=true "Model deployment")


Deployment takes some time to conclude, but when it finishes successfully the ACI web service has a status of ***Healthy*** and the model is deployed correctly. We can now move to the next step of actually testing the endpoint.
### Consuming/testing the endpoint (ACI service)

_Endpoint (Azure Machine Learning Studio)_

![ACI service](images/18.png?raw=true "ACI service")

After the successful deployment of the model and with a _Healthy_ service, I can print the _scoring URI_, the _Swagger URI_ and the _primary authentication key_:

![ACI service status and data](images/22.png?raw=true "ACI service status and data")


The scoring URI can be used by clients to submit requests to the service.

In order to test the deployed model, we used a _Python_ file, named `endpoint.py`:

![endpoint.py file](images/18.png?raw=true "endpoint.py file")

In the beginning, we fill in the `scoring_uri` and `key` with the data of the _aciservice_ printed above. We can test our deployed service, using test data in JSON format, to make sure the web service returns a result.

In order to request data, the REST API expects the body of the request to be a JSON document with the following structure: 

```
{
    "data":
        [
            <model-specific-data-structure>
        ]
}
```

In order to test the deployed service, one could use the above file by inserting data in the `endpoint.py` file, saving it, and then run the relevant cell in the `automl.ipynb` Jupyter Notebook.

**Another way** would be using the Swagger URI of the deployed service and the [Swagger UI](https://swagger.io/tools/swagger-ui/).

**A third way** would also be to use Azure Machine Learning Studio. Go to the _Endpoints_ section, choose _aciservice_ and click on the tab _Test_:

![Testing ACI service in Azure ML Studio](images/29.png?raw=true "Testing ACI service in Azure ML Studio")

Fill in the empty fields with the medical data you want to get a prediction for and click _Test_:


## Flask Deployment
  Code:
    ![Flask Deployment Code](images/41.jpg?raw=true "Flask Deployment Code")
    ![Flask Deployment Code](images/42.jpg?raw=true "Flask Deployment Code")
    ![Flask Deployment Code](images/43.jpg?raw=true "Flask Deployment Code")


## Front-End of the Project
  For the front-end of the project, the primary objective was to develop an interactive website that
  can collect relevant data from the users and provide information and present an accurate
  prediction based on the data given while keeping the website simple and free of confusion to
  improve user comfort and experience.

  The landing page is a simple and brief description of what the website is about. The top most
  header area contains basic contact information along with some social media links.
  
  ![Landing Page](images/44.jpg?raw=true "Landing Page")

  
  The section ‘Predictor’ is where the main deployed model is consumed. In other words, the form
  in this section is the main tool in gaining the user’s data and predicting it.
  
  The form takes values of all the different attributes that are used in the model building process
  (For ex: Age, Diabetes, Smoking,Creatinine Phosphokinase etc.) and sends it to the python
  script where the prediction is made and the output is sent back.
  
  ![Predictor Form](images/45.jpg?raw=true "Predictor Form")


## Chatbot Integration

Along with all the other features and functionalities mentioned about the site, a chatbot is also
integrated for better user experience.

This chatbot is integrated with chatGPT which is a huge trend these days.
Integrating ChatGPT with social intents involves leveraging the capabilities of the language
model to generate responses that are appropriate and contextually relevant to different types of
social interactions.

![Chatbot](images/46.jpg?raw=true "Chatbot")


## Screen Recording

The screen recording can be found [here](https://youtu.be/qA46qczT0ak) and it shows the project in action. 

More specifically, the screencast demonstrates:

- A working model
- Demo of the deployed  model
- Demo of a request sent using a Web Application


## Dataset Citation

Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. [BMC Medical Informatics and Decision Making 20, 16 (2020)](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5).


## References

- Research article: [Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)
- [Heart Failure Prediction Dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)
- [Consume an Azure Machine Learning model deployed as a web service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?tabs=python)
- [Deploy machine learning models to Azure](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azcli)
- [A Review of Azure Automated Machine Learning (AutoML)](https://medium.com/microsoftazure/a-review-of-azure-automated-machine-learning-automl-5d2f98512406)
- [The Holy Bible of Azure Machine Learning Service. A walk-through for the believer (Part 3)](https://santiagof.medium.com/the-holy-bible-of-azure-machine-learning-service-a-walk-through-for-the-believer-part-3-74fb7393fffc)
- [What is Azure Container Instances (ACI)?](https://docs.microsoft.com/en-us/azure/container-instances/container-instances-overview)
- [AutoMLConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py)
- [Using Azure Machine Learning for Hyperparameter Optimization](https://dev.to/azure/using-azure-machine-learning-for-hyperparameter-optimization-3kgj)
- [hyperdrive Package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py)
- [Tune hyperparameters for your model with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
- [Configure automated ML experiments in Python](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train)
- [How Azure Machine Learning works: Architecture and concepts](https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-architecture)
- [Configure data splits and cross-validation in automated machine learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cross-validation-data-splits)
- [How Much Training Data is Required for Machine Learning?](https://machinelearningmastery.com/much-training-data-required-machine-learning/)
- [How Can Machine Learning be Reliable When the Sample is Adequate for Only One Feature?](https://www.fharrell.com/post/ml-sample-size/)
- [Modern modelling techniques are data hungry: a simulation study for predicting dichotomous endpoints](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-14-137)
- [Predicting sample size required for classification performance](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-12-8)
