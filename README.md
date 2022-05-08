# Project 1: Content Based Recommender

### Project description
The goal of this project is to prepare data from real world hotel dataset, and use the Content-based recommender to train on this set so it gives the best recommendations. It is measured by achieving the best HR@10 score in the final evaluation.

The project constists of two jupyter notebooks:
- `project_1_data_preparation.ipynb` - showing how the data is prepared to be used later in the reccomender
- `project_1_recommender_and_evaluation.ipynb` - preparing, tuning and testing the reccomender

These are the reccomenders used and their scores on hotel dataset:
- `LinearRegressionCBUIRecommender`
![image](https://user-images.githubusercontent.com/96208361/167309696-e0bcc83b-2beb-4b9b-9ca8-b568f6062379.png)
- `RandomForestCBUIRecommender` 
![image](https://user-images.githubusercontent.com/96208361/167309433-3653d097-d3c6-4e87-b292-edc93484f6d8.png)
- `XGBoostCBUIRecommender`
![image](https://user-images.githubusercontent.com/96208361/167309534-47aeadf5-b962-407b-93f3-47f49e2c6127.png)

The best scores achieved in comparison to the Amazon reccomender:
![image](https://user-images.githubusercontent.com/96208361/167309851-c3d4f775-b80f-44c7-9357-361451523ebc.png)



### Requirements to run the notebook
#### To run this project Python3 environment and following steps are required:

1.  Installing jupyter notebook environment:

		pip install notebook
2.  Installing all packages (at once) needed to run notebook succesfully:

		pip install numpy pandas seaborn matplotlib sklearn hyperopt IPython
3.  Running jupyter notebook (from commandline in the project folder destination):

		jupyter notebook


