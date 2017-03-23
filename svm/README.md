## Installation

Install numpy
```{r, engine='bash', instNumpy}
pip install -U numpy[alldeps]
```
Install scipy
```{r, engine='bash', instScipy}
pip install -U scipy[alldeps]
```
Install scikit-learn
```{r, engine='bash', instScikitLearn}
pip install -U scikit-learn[alldeps]
```
You can install matplotlib, if required.
```{r, engine='bash', instMatplot}
pip install matplotlib
```

## Usage

Just run the following to **train** and **test** the SVM models.
```{r, engine='bash', command}
python main.py
```
Note: Change the value of `noOfTotalClasses`, `noOfTrainingVectors` and
`noOfTestingVectors` as required during execution.
