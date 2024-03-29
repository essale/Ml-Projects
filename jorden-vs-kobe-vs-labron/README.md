# Summary:
Here we will analyze and understand the differences between 
* Kobe Bryant 
* Lebron James
* Michael Jordan

<center><img 
src = "https://i.imgur.com/mebnxNv.jpg">
</center>

****
## Data Set
* Source: Keras
* The data sets containing Michael Jordan, Kobe Bryant and Lebron James stats.

Let’s get an idea of what we’re working with.

****
#### General Terms
<center>

|   **Term**    |    Meaning                                  |  **Term**     |  Meaning                          |
|:--------------|:--------------------------------------------|:--------------|:----------------------------------|
| **Pos**       | Position                                    | **eFG%**      | Effective Field Goal Percentage   |
| **Tm**        | Team                                        | **Lg**        | League                            |
| **G**         | Games                                       | **+/-**       | Plus/Minus                        |
| **GS**        | Games Started                               | **GmSc**      | Game Score                        | 
| **MP**        | Minutes Played Per Game                     | **PTS**       | Points per Game                   |
| **FG**        | Field Goals Per Game                        | **PF**        | Personal Fouls Per Game           |
| **FGA**       | Field Goal Attempts Per Game                | **TOV**       | Turnovers Per Game                |
| **FG%**       | Field Goal Percentage                       | **BLK**       | Blocks Per Game                   |
| **3P**        | 3-Point Field Goals Per Game                | **STL**       | Steals Per Game                   |
| **3PA**       | 3-Point Field Goal Attempts Per Game        | **AST**       | Assists Per Game                  |
| **3P%**       | 3-Point Field Goal Percentage               | **TRB**       | Total Rebounds Per Game           |
| **2P**        | 2-Point Field Goals Per Game                | **DRB**       | Defensive Rebounds Per Game       |
| **2PA**       | 2-Point Field Goal Attempts Per Game        | **ORB**       | Offensive Rebounds Per Game       |
| **2P%**       | 2-Point Field Goal Percentage               | **FT%**       | Free Throw Percentage             |
| **FT**        | Free Throws Per Game                        | **FTA**       | Free Throw Attempts Per Game      |
</center> 

<center>

|   **Term**    |   **Meaning**                      |   **Exlanation**                      |
|:--------------|:-----------------------------------|:--------------------------------------|
| **PER**       | Player Efficiency Rating           |A measure of per-minute production standardized such that the league average is 15.|
| **TS%**       | True Shooting Percentage           |A measure of shooting efficiency that takes into account 2-point field goals, 3-point field goals, and free throws.|
| **3PAr**      | 3-Point Attempt Rate               |Percentage of FG Attempts from 3-Point Range.|
| **FTr**       | Free Throw Attempt Rate            |Number of FT Attempts Per FG Attempt.|
| **ORB%**      | Offensive Rebound Percentage       |An estimate of the percentage of available offensive rebounds a player grabbed while he was on the floor.|
| **DRלB%**      | Defensive Rebound Percentage       |An estimate of the percentage of available defensive rebounds a player grabbed while he was on the floor.|
| **TRB%**      | Total Rebound Percentage           |An estimate of the percentage of available rebounds a player grabbed while he was on the floor.|
| **AST%**      | Assist Percentage                  |An estimate of the percentage of teammate field goals a player assisted while he was on the floor.|
| **STL%**      | Steal Percentage                   |An estimate of the percentage of opponent possessions that end with a steal by the player while he was on the floor.|
| **BLK%**      | Block Percentage                   |An estimate of the percentage of opponent two-point field goal attempts blocked by the player while he was on the floor.|
| **TOV%**      | Turnover Percentage                |An estimate of turnovers committed per 100 plays.|
| **USG%**      | Usage Percentage                   |An estimate of the percentage of team plays used by a player while he was on the floor.|
| **OWS**       | Offensive Win Shares               |An estimate of the number of wins contributed by a player due to his offense.|
| **DWS**       | Defensive Win Shares               |An estimate of the number of wins contributed by a player due to his defense.|
| **WS**        | Win Shares                         |An estimate of the number of wins contributed by a player.|
| **WS/48**     | Win Shares Per 48 Minutes          |An estimate of the number of wins contributed by a player per 48 minutes (league average is approximately .100)|
| **OBPM**      | Offensive Box Plus/Minus           |A box score estimate of the offensive points per 100 possessions a player contributed above a league-average player, translated to an average team.|
| **DBPM**      | Defensive Box Plus/Minus           |A box score estimate of the defensive points per 100 possessions a player contributed above a league-average player, translated to an average team.|
| **BPM**       | Box Plus/Minus                     |A box score estimate of the points per 100 possessions a player contributed above a league-average player, translated to an average team.|
| **VORP**      | Value over Replacement Player      |A box score estimate of the points per 100 TEAM possessions that a player contributed above a replacement-level (-2.0) player, translated to an average team and prorated to an 82-game season.|
 
</center> 

****
#### Hypothesis
 * Which player (Labron, Kobe, Jordan) is statistically better 


# Installations:
* Analyze: Pandas, numpy, jupyter
* Visualize: matplotlib, seaborn
* ML: sklearn

# Data sources:
* [Kaggle](https://www.kaggle.com/xvivancos/michael-jordan-vs-kobe-bryant-vs-lebron-james/code)
* [Classification modules](https://medium.com/@Mandysidana/machine-learning-types-of-classification-9497bd4f2e14)

# Explanations:
* Confusion Matrix 
[Confusion Matrix](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/)
[ConfusionMatrix 2](https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019)
```
A confusion matrix is a table that is often used to describe the performance of a classification model (or “classifier”) on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.
```
* Logistic Regression 
[Logistic Regression](https://towardsdatascience.com/logistic-regression-classifier-8583e0c3cf9)
```
Logistic Regression is a ‘Statistical Learning’ technique categorized in ‘Supervised’ Machine Learning (ML) methods dedicated to ‘Classification’ tasks.
```

* Naive Bayes
[naive-bayes](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)
[naive-bayes2](https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn)

* Decision Trees
[Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
```
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
```