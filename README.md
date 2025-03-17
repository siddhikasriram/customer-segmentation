# Customer Segmentation Analysis: K-Means Clustering

## Introduction

In this report, we present an analysis of customer data from an electronic retailer with the goal of clustering customers based on three key attributes: visits, number of items bought, and spending amount.

---

## Data Preprocessing

The dataset used for this analysis is `visitItemSpend.csv`. The data required certain preprocessing steps due to the presence of outliers and negative values in the attributes. To prevent attributes with high magnitudes, such as spending, from dominating the clustering process, we employed **Min-Max scaling**. This technique normalizes all attributes within a range of 0–1, ensuring that all attributes carry equal weight in the clustering process.

---

## Cluster Analysis

We used the **K-Means** algorithm for performing the cluster analysis. This clustering process identified distinct customer segments based on three attributes: visits, number of items bought, and normalized spending. The clusters can be mapped to four customer types:

1. **Semi-loyal, big spender**
2. **Loyal, moderate spender**
3. **Semi-loyal, moderate spender**
4. **Infrequent, low spender**

### Cluster Means

The cluster means represent the average values for each of the four clusters, based on the attributes of visits, number of items purchased, and spending. These means serve as the defining characteristics of each customer segment.

---

## Cluster Labels

We assigned labels to the four clusters based on the patterns of customer visits and spending. To determine these labels, we examined the mean values for visits and spending. A higher value in either of these columns indicates more frequent visits or higher spending.

- **Semi-loyal, big spender** is assigned to cluster 4.
- **Loyal, moderate spender** is assigned to cluster 1.
- **Semi-loyal, moderate spender** is assigned to cluster 2.
- **Infrequent, low spender** is assigned to cluster 3.

---

## Number of Customers Belonging to Each Cluster

After fitting the model, we predicted the cluster each customer belongs to. We counted the number of customers in each cluster using a dictionary, which showed that:

- **Cluster 3**, containing customers with infrequent visits and low spending patterns, had the highest number of customers.

---

## Analysis and Results

The analysis of the clusters provides valuable insights into customer behavior:

- **Customer Segmentation:** Cluster analysis helps the business owner understand their customer base better by identifying distinct customer segments. This segmentation allows the owner to tailor marketing strategies, product offerings, and customer service to meet the needs and preferences of each segment.
  
- **Pricing Strategies:** Different customer segments may have varying price sensitivities. By understanding the spending behavior of each cluster, the business owner can adjust pricing strategies to maximize revenue and profitability while remaining competitive.

- **Customer Retention:** Identifying loyal customer segments allows the owner to implement strategies to retain valuable customers. Personalized loyalty programs, exclusive offers, and improved customer support can enhance customer retention.
