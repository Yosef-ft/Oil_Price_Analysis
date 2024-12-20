# Oil Price Analysis

## Overview
The objective of this analysis is to evaluate the impact of major global events—such as political decisions, regional conflicts, economic sanctions, and OPEC policies—on Brent oil prices. Through a structured analysis of time-series data and statistical modeling, this project aims to extract data-driven insights that can enhance decision-making for investors, policymakers, and energy companies.

## Project Goals
1. Identify Key Events Impacting Oil Prices: Evaluate historical events that have shown substantial influence on Brent oil prices over the past decade.
2. Quantify Price Response to Events: Measure how much oil prices change in response to specific event types.
3. Provide Actionable Insights: Derive recommendations for stakeholders by connecting data insights with strategic needs.

## Getting Started
### Prerequisites
Make sure you have the following installed:
  * Python 3.x
  * Pip (Python package manager)

### Installation
Clone the repository:
```
git clone https://github.com/Yosef-ft/Oil_Price_Analysis.git
cd Oil_Price_Analysis
```
Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the required packages:
```
pip install -r requirements.txt
```

Training LSTM model
```
dvc repro
```

## Pipeline DAG
```mermaid
flowchart TD
	node1["data_stage"]
	node2["feature_stage"]
	node1-->node2
	node3["evaluate_stage"]
  
	node4["report_stage"]
	node5["train_stage"]
  node2-->node5
	node5-->node3
	node5-->node4

```