# SmartCart Customer Clustering

## Overview
This project focuses on customer segmentation for SmartCart, an e-commerce platform, using unsupervised machine learning techniques. By clustering customers based on their demographic and behavioral data, we aim to identify distinct customer groups to enable targeted marketing strategies, personalized recommendations, and improved customer retention.

The analysis leverages data preprocessing, feature engineering, dimensionality reduction, and clustering algorithms to uncover patterns in customer behavior, spending habits, and demographics.

## Dataset
The dataset (`smartcart_customers.csv`) contains customer information including:
- Demographic details: Age, Education, Marital Status, Income
- Behavioral data: Recency (days since last purchase), Response to campaigns
- Spending categories: Wines, Fruits, Meat Products, Fish Products, Sweet Products, Gold Products
- Family information: Number of children (kids and teens)
- Enrollment date: Date of customer enrollment

**Dataset Size**: 2240 rows × 22 columns

## Technologies and Libraries
- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning algorithms (Preprocessing, Clustering, PCA, Metrics)
- **Kneed**: Elbow point detection for optimal K

## Methodology
The project follows a comprehensive machine learning pipeline for customer clustering:

### 1. Data Loading and Exploration
- Load the dataset using Pandas
- Initial inspection: `df.head()`, `df.info()`, `df.shape`
- Check for missing values: `df.isnull().sum()`

### 2. Data Preprocessing
- **Column Standardization**: Convert column names to lowercase
- **Handling Missing Values**: Impute missing income values with median
- **Feature Engineering**:
  - Customer Tenure: Calculate days since enrollment (`dt_customer`)
  - Age: Derive from year of birth (using 2026 as reference year)
  - Total Spending: Sum of all product category spendings
  - Education Mapping: Group education levels (e.g., Graduation → Graduate, PhD → PostGraduate)
  - Marital Status Mapping: Simplify to "Partner" or "Alone"
  - Total Children: Sum of kidhome and teenhome

### 3. Data Cleaning
- **Outlier Removal**: Filter out extreme values (age < 100, income < 600,000)
- **Column Dropping**: Remove irrelevant columns (ID, original spending columns, etc.) to create a cleaned dataset

### 4. Exploratory Data Analysis (EDA)
- **Pair Plot**: Visualize relationships between key numerical features
- **Correlation Heatmap**: Analyze correlations between variables using Seaborn

### 5. Feature Encoding
- **One-Hot Encoding**: Convert categorical features (Education, Living With) to numerical using `OneHotEncoder`

### 6. Feature Scaling
- **Standardization**: Apply `StandardScaler` to normalize features for clustering algorithms

### 7. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Reduce to 3 principal components for visualization and clustering efficiency
- Explained variance ratio analysis

### 8. Determining Optimal Number of Clusters (K)
- **Elbow Method**: Calculate Within-Cluster Sum of Squares (WCSS) for K=1 to 10
- **Silhouette Score**: Evaluate clustering quality for K=2 to 10
- **KneeLocator**: Automatically detect elbow point for optimal K (found to be 4)

### 9. Clustering Algorithms
- **K-Means Clustering**: Apply with K=4 clusters, using PCA-transformed data
- **Agglomerative Clustering**: Hierarchical clustering with Ward linkage and K=4 clusters

### 10. Visualization
- **3D Scatter Plots**: Visualize clusters in PCA space for both algorithms
- **Combined Plot**: Overlay WCSS and Silhouette Scores for K selection
- **Cluster Characterization**: Count plot, scatter plot of spending vs. income by cluster

### 11. Cluster Analysis
- **Cluster Summary**: Compute mean values for each cluster across features
- **Insights**: Identify patterns in spending, income, demographics per cluster

## Results and Insights
- **Optimal Clusters**: 4 clusters identified using Elbow method and Silhouette analysis
- **Cluster Profiles** (based on analysis):
  - Cluster 0: [Describe based on summary, e.g., High spenders with high income]
  - Cluster 1: [e.g., Moderate spenders, middle income]
  - Cluster 2: [e.g., Low spenders, younger customers]
  - Cluster 3: [e.g., High recency, low engagement]
- **Key Findings**: Correlations between income and total spending, impact of family size on purchasing behavior, etc.

## How to Run the Project
1. **Prerequisites**:
   - Python 3.x installed
   - Required libraries: Install via `pip install pandas numpy matplotlib seaborn scikit-learn kneed`

2. **Clone/Download**: Place the notebook and dataset in the same directory

3. **Run the Notebook**:
   - Open `smartcart.ipynb` in Jupyter Notebook or VS Code
   - Execute cells sequentially to reproduce the analysis

4. **Dataset**: Ensure `smartcart_customers.csv` is in the project directory

## File Structure
```
SmartCart Customer Clustering/
├── smartcart.ipynb          # Main analysis notebook
├── smartcart_customers.csv  # Dataset
└── README.md                 # This file
```

## Future Improvements
- Experiment with other clustering algorithms (e.g., DBSCAN, Gaussian Mixture Models)
- Incorporate additional features or external data
- Deploy as a web app for interactive cluster exploration
- Validate clusters with supervised learning on labeled data

## 👨‍💻 Author  

**Prince Jha**  
- 🐙 GitHub: [@princejha-dev](https://github.com/princejha-dev)  
- 💼 LinkedIn: [princejha-dev](https://linkedin.com/in/princejha-dev)

## License
This project is for educational purposes. Dataset and code are provided as-is.