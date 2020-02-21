# Coding Challenge - Data Scientist

This repository was created to host the helper files for the Data Scientist role challenge.

The instructions below imply some degree of knowledge of an extensive set of skills related to professional Data Science \
work. Unless you are applying for a senior position, it is OK if you are not familiar with some of those skills (E.g. using\
git or integrating models with Python production code) - just focus on deliverying the pieces you know, within a reasonable\
time, like you would normally do at work. It is also OK to ask for clarifications or prioritization guidance.

## The problem - Consultation finishing time estimation at a clinic

On this task you will have to create a model to predict the time a patient will leave a clinic, after having a \
consultation with a doctor, in an appointment-free service.

The clinic receives patients from 14:00 (time 0) to 18:00. Patients that arrived before 18:00 wait inside the clinic \
for their consultation.

There is a triage room with capacity for one patient at a time. As soon as it is available, arriving patients will pass \
a quick assessment to flag the patients that should have priority in seeing a doctor. At the end of this assessment, \
patients should be given an estimate of the time they will be free, i.e., the expected time for finishing their \
consultation. Your job is to create a system to give that estimate.

After the assessment, patients will wait until any of the 6 doctors working at the clinic are free. If none is free, \
patients will wait in a FIFO virtual queue. If a a new patient receives an `urgent` assessment, she will wait in the \
`urgent` FIFO virtual queue. Vacant doctors will always first receive patients from that queue. At the moment they call \
a new patient, doctors can see the size of both queues.

Consultations don't have a fixed duration. Instead, doctors are supposed to give patients the attention demanded by their \
cases.

The data provided refers to a continuous sequence of working days. Your model will be evaluated against the following 5 \
days. For your convenience, a csv summarizing data in a tabular format is provided. That file is intended as a shortcut for\
your first exploratory analises. That file can also be enriched with new features and used to train your model if you wish \
so.

### Details

For this task you can use any language of your choice for analysis and modeling. However you will need to integrate your \
model with the provided evaluation code in Python 3.

You will need to:
* Create a local git repository
* Download the Python files provided: `evaluate.py`, `file_evaluation.py` and `your_model.py` 
* Install Python 3 in a virtual environment
* Download the dataset for this task from your e-mail. (__Note:__ There is no limit on the time you need to return your \
solution. However that time will be taken into consideration when analyzing your solution.)
* Create a program to (re-)train and serialize a model to solve the problem.
* Create a program to use the trained model in the evaluation of additional data. (Integrate it in `your_model.py`)
* Submit your repo and your presentation within the deadline

### Procedure

1. Start a git repository with ```git init```
1. Install Python 3 and create a virtual env
1. Do your magic on your local machine, trying to commit often
1. Add your README, with instructions on how to run your code
1. Run ```git bundle create YOURNAME.bundle HEAD ```
1. Send the generated file by email back to your interviewer

### Evaluation tool usage

1. Install Python 3 and cd to the solution folder
1. Run ```virtualenv env```
1. Run ```. env/bin/activate``` (commands might differ depending on your OS)
1. Run ```pip3 install -r requirements.txt```
1. Create a subfolder and move the csv files for the days you want to evaluate
1. Run ```python3 evaluate.py SUBFOLDER_NAME```

## Presentation

We want you to create a small presentation (use Google Slides) summarizing your findings and answering a few questions. \
However the slides should be self-explanatory, be prepared to present your slides and answer additional questions.

Your findings should focus on any relevant facts about the problem or concerns about the quality of the service provided\
by the clinic.

The questions we want you to answer in your presentation are:
* Is it better to break down the problem into the forecasting of different components? Why?
* What is the relationship between patient temperature and consultation duration?
* Is there a relationship between time of the day and consultation duration? What could likely explain that?
* Apart from the assessment result (_normal_, _urgent_), what is the feature that is more relevant for the problem?
* What is your opinion about the scoring metric implemented in `evaluate.py`? Is there real-world support for it?
* If you had more days to work on the solution, how would you architect the solution?

## Evaluation criteria

Things we like:
* __Code readability, documentation and testing__ - We like code that is easy to read. PEP8 is a good start. Names for \
functions and variables should be self-explanatory. Documentation is good, but it is even better when it is not \
needed. Unit tests and doc-tests should be commonplace in production code, and can sometimes improve the reliability of \
analysis code. Same-order performance improvements are nice, but code simplicity is even nicer.
* __Understanding of the problem__ - Models should be build with a clear objective in mind. Evaluation metrics should be \
aligned with the objectives of the model. Understanding the real-life mechanics of the phenomena being modeled is crucial \
to coming up with the right approach.
* __Discovering THE features__ - Sometimes the features that really matter are hidden within the data. Spending time \
analyzing the data, treating and manipulating it, can reveal the information key to solve the problem.

We want you to:
* Analyze the data with care
* Create a small presentation about your findings, and answering our specific questions
* Train a prediction model and integrate it with our evaluation code
* Submit the repo with your working history

You don't need to:
* Try several different models to try to improve your score by subdecimal points
* Over-document or spend too much time writing tests
* Keep a history of several variations of each of your analysis until you got the good ones
* Spend too much time overdoing it
* Spend too little time and miss the important findings for the solution 
