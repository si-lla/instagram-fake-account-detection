# 📷 Instagram Fake vs Genuine Account Detection

This project uses **machine learning** to classify Instagram accounts as **fake (spammer)** or **genuine**, based on user profile characteristics. The results are visualized using an interactive **Power BI dashboard**.

---

## 🧠 Project Objective

To identify behavioral patterns in Instagram user profiles and train a machine learning model to detect fake or spam accounts with high accuracy.

---

## 📁 Dataset

- `train.csv`: Labeled training dataset (`1` = fake, `0` = genuine)
- `test.csv`: Used for model evaluation
- Source: Collected via web scraping (March 2019)

---

## 🔧 Tools & Technologies

- 🐍 Python – pandas, numpy, scikit-learn, seaborn
- 📊 Power BI – interactive visual dashboard
- 💻 VS Code – development environment

---

## 📈 Features Used

- `profile pic`: Profile picture presence  
- `#followers`, `#follows`, `#posts`: Engagement metrics  
- `private`, `external URL`: Privacy and link behavior  
- `description length`, `fullname length`: Text-based features

---

## 🤖 ML Model Summary

- **Algorithm**: Decision Tree Classifier  
- **Train/Test Split**: 80/20  
- **Accuracy Achieved**: **93%**  
- **Balanced Precision/Recall**: 0.93  
- **Key Predictive Features**: `profile pic`, `#followers`, `#follows`, `description length`

---

## 📊 Power BI Dashboard

The dashboard provides a comprehensive view of:

- 📌 Account Type Distribution (Fake vs Genuine)  
- 📌 Bio Length Comparison  
- 📌 Engagement Metrics (Followers, Posts, Follows)  
- 📌 Privacy & Profile Feature Comparison  
- 📌 Gauge for ML Model Accuracy  
- 📌 Slicer for filtering account types  

📷 **Dashboard Preview:**

![Dashboard Screenshot](https://github.com/si-lla/instagram-fake-account-detection/blob/main/final-dashboard.png?raw=true)


---

## 🧪 ML Visual Outputs

## 🧪 ML Visual Outputs

Below are key visualizations generated during the data exploration and machine learning workflow:


- ![Dataset Overview](https://github.com/si-lla/instagram-fake-account-detection/blob/main/dataset-overview.png?raw=true)
- ![Correlation Heatmap](https://github.com/si-lla/instagram-fake-account-detection/blob/main/correlation-heatmap.png?raw=true)
- ![Followers Count Comparison](https://github.com/si-lla/instagram-fake-account-detection/blob/main/followers-count-comparison.png?raw=true)
- ![Following Count Comparison](https://github.com/si-lla/instagram-fake-account-detection/blob/main/following-count-comparison.png?raw=true)
- ![Posts Count Comparison](https://github.com/si-lla/instagram-fake-account-detection/blob/main/posts-count-comparison.png?raw=true)
- ![Profile Picture Presence](https://github.com/si-lla/instagram-fake-account-detection/blob/main/profile-pic-presence.png?raw=true)
- ![Feature Importance](https://github.com/si-lla/instagram-fake-account-detection/blob/main/feature-importance.png?raw=true)
- ![Decision Tree Visualization](https://github.com/si-lla/instagram-fake-account-detection/blob/main/decision-tree-visualization.png?raw=true)
- ![Confusion Matrix](https://github.com/si-lla/instagram-fake-account-detection/blob/main/confusion-matrix.png?raw=true)
---

## 📂 Project Structure

| File / Folder              | Description                            |
|---------------------------|----------------------------------------|
| `main.py`                 | Python script for EDA and ML training  |
| `train.csv`, `test.csv`   | Input datasets                         |
| `tableau_summary.csv`     | Cleaned summary data for visualization |
| `dashboard.pbix`          | Power BI dashboard file                |
| `assets/`                 | Dashboard + ML output screenshots      |
| `README.md`               | Project documentation (this file)

---

## 🚀 How to Run

1. Clone this repository  
2. Run `main.py` to train and evaluate the model  
3. Open `tableau_summary.csv` in Power BI  
4. Use `dashboard.pbix` to view or edit the dashboard  

---

## 🧑‍💼 Author

- **Name**: *Silla Shaju*  
- **Internship**: Unified Mentor Pvt. Ltd.  
- **Project**: Instagram Fake, Spammer & Genuine Account Detection

---

## 📬 Contact

- 💼 [LinkedIn](https://www.linkedin.com/in/silla-shaju-309b66322?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)  
- 💻 [GitHub](https://github.com/si-lla)

---

