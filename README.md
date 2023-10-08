# lab-kaggle-competition

![portada](https://github.com/Ironhack-Data-Madrid-Enero-2021/W7-Kaggle_competition/blob/main/images/PORTADA.jpg)

## Description

- Find the best machine learning model and params for a given dataset. 

## Instructions

Find the Kaggle competition with your cohort name, i.e. **diamonds-databcn0722**, https://www.kaggle.com/competitions/diamonds-databcn0722/overview
### train.csv
* 1. **Processing/cleaning** the dataset: this should be later modularized in functions.
* 2. **Train** a model (fit & predict) with the data in `train.csv`. This file DOES contain a **y** (price).
        - Do *train, test, split* on `train.csv` if necessary.
        - Choose the best model regarding the metrics. In this case, the lowest RMSE (error).

        2.1. **Export** the model: we don't want to invest time/RAM resources on training the model again in the future.

### test.csv
* 3. Apply the same **cleaning** to `test.csv`. This files does NOT contain a **y** (no price column).
* 4. We'll apply the already **trained model** from step 2 to the `test.csv` file. With this we'll generate a new column with the predicted values.  

### my_submission.csv
* 5. Generate a `submission.csv` file with only two columns: the **ID** of the diamond & the predicted **price** (y).

In other words: use `train.csv` to generate and save a model. Use `test.csv` to predict new values. Then generate a DF with ID & predicted. 

## Deliverables

- **Jupyter notebooks** where you show the process you followed to get to your submissions.

- A **slide** (.ppt, ipynb, etc) with a summary of the metrics you obtained and the rationale behind it. 
    - Why do those params work better than others?

## Tips
- Check often for the df.shape & len of the things your working with
- Do make sure you ONLY have two columns. When saving the file, you might save an "extra column" (index). So make sure you don't include it. There should only be two columns: id & price
- Take advantage of the daily submissions. Try at least one today!
