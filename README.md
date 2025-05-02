# Iris Dataset Analysis

This project demonstrates basic data analysis and visualization techniques using the Iris dataset. The Iris dataset is a classic dataset in machine learning and statistics, containing measurements of sepal and petal lengths and widths for three species of iris: setosa, versicolor, and virginica.

## Tasks Performed

1.  **Load and Explore the Dataset:**
    * Loaded the Iris dataset using pandas.
    * Displayed the first few rows of the dataset to inspect the data.
    * Explored the structure of the dataset, including data types and checking for missing values.
    * Cleaned the dataset (although the Iris dataset is inherently clean and contains no missing values).

2.  **Basic Data Analysis:**
    * Computed basic statistics of the numerical columns (mean, standard deviation, min, max, quartiles).
    * Performed groupings on the 'target' column (representing the Iris species) and computed the mean of the sepal and petal measurements for each species.
    * Identified initial patterns and findings, such as the variation in petal dimensions across different species.

3.  **Data Visualization:**
    * Created a line chart (demonstrating syntax, though not ideally suited for this specific analysis).
    * Generated a bar chart showing the average petal length for each Iris species.
    * Plotted a histogram to visualize the distribution of sepal width.
    * Created a scatter plot to explore the relationship between sepal length and petal length, colored by species.
    * Customized all plots with titles, axis labels, and legends for clarity.

## Libraries Used

* **pandas:** For data manipulation and analysis.
* **numpy:** For numerical operations.
* **matplotlib:** For creating basic plots.
* **seaborn:** For enhanced and statistically-oriented visualizations.
* **sklearn.datasets:** To load the built-in Iris dataset.

## Setup

To run this project, you will need to have Python installed on your system along with the libraries listed above. You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
