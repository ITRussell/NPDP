# Report

## Abstract 
This technical memo examines the feasibility of applying machine learning techniques for anomaly detection on encrypted network data using Zeek logs. It compares the effectiveness of various machine learning approaches with an alternative commercial solution, ThatDot Novelty Detector. The UNSW-NB15 dataset was selected for testing and is a comprehensive network traffic dataset designed to evaluate the performance of intrusion detection systems (IDS) and other network security solutions. The experiment aimed to identify the best features for predicting malicious behavior between two endpoints in a network using the UNSW-NB15 datasets and evaluate the performance of traditional machine learning models with a commercial grade tool.  
 
## Introduction 

In today's rapidly evolving digital landscape, network security has become an essential aspect of any organization's IT infrastructure. One critical aspect of network security is the detection of anomalies, which are deviations from normal traffic patterns that may indicate potential threats such as intrusions, malware, and unauthorized access. Anomaly detection helps identify and mitigate these threats, thereby preventing the compromise of sensitive data and the disruption of critical services. 

However, with the growing prevalence of encrypted network data, detecting anomalies has become increasingly challenging. Encryption provides robust protection for sensitive information, but it also limits the visibility of network traffic patterns, making it difficult for traditional intrusion detection systems to identify malicious activities. As a result, there is a need for novel approaches to detect anomalies in encrypted network data without compromising privacy and security. 

Zeek, formerly known as Bro, is a powerful open-source network security monitoring tool that can be used to generate comprehensive logs of network activities. Zeek logs contain valuable information on various network events, such as connections, DNS requests, and HTTP transactions. By analyzing these logs, security professionals can gain insight into network behavior and identify potential threats. 

The objective of this technical memo is to assess the feasibility of applying contemporary machine learning techniques for anomaly detection on encrypted network data using Zeek logs. We will explore various machine learning approaches, evaluate their effectiveness in detecting anomalies in encrypted network data, and compare them with an alternative commercial solution such as the ThatDot Novelty Detector. Through this investigation, we aim to provide practical recommendations for enhancing network security monitoring in the face of encrypted data challenges. 

## Overview of contemporary machine learning techniques and commercial tools 

Contemporary machine learning techniques encompass a wide range of approaches, from traditional supervised and unsupervised learning methods to advanced deep learning techniques, each presenting its own set of challenges when dealing with encrypted data. Supervised learning methods, such as Support Vector Machines (SVM), Decision Trees, Random Forests, and Neural Networks, rely on labeled data for training and prediction. In the context of encrypted data, acquiring sufficient labeled data for effective training can be difficult, as sensitive information is concealed, limiting the availability of useful features for training. 

Unsupervised learning methods, such as clustering algorithms (e.g., K-means, DBSCAN) and dimensionality reduction techniques (e.g., PCA), are used to identify patterns and structures in unlabeled data. While these methods can potentially uncover hidden patterns within encrypted data, their reliance on data distributions and feature spaces may be hindered by the obfuscation introduced by encryption. This can result in poor performance or increased false positives when dealing with encrypted network traffic. 

Semi-supervised learning, which combines both supervised and unsupervised learning techniques, may alleviate some challenges posed by encrypted data, but it still requires a certain level of labeled data, which may be difficult to obtain. Deep learning methods, including Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), leverage complex architectures to model high-dimensional and sequential data. While these methods have shown promise in various domains, their applicability to encrypted data is limited due to the inherent lack of data visibility and the need for substantial computational resources. 

In addition to these techniques, there are commercially available tools like ThatDot Novelty Detector, which provides a unique solution for real-time anomaly detection on categorical data, even in the context of encrypted data. It uses a context-based approach to assess anomalies, dramatically reducing false positives by learning contextual information automatically from the data. This feature is particularly valuable when dealing with encrypted data, as the limited available information can lead to higher false positive rates in traditional anomaly detection methods. 

ThatDot Novelty Detector builds and maintains a dynamic graphical model structured by the input data, which represents a complex set of conditional probabilities for each component of the data observation. In the case of encrypted data, this dynamic model allows the system to continuously adapt to changing data patterns resulting from evolving encryption techniques or variations in network traffic. This enables the system to effectively identify truly anomalous observations while minimizing false positives, making it a valuable commercial tool for network security monitoring and anomaly detection, even when faced with the challenges of encrypted data. 

### Data 

The UNSW-NB15 dataset is a comprehensive network traffic dataset designed to evaluate the performance of intrusion detection systems (IDS) and other network security solutions. It contains a mix of normal traffic and various attack types, including Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms, simulating real-world network environments. The dataset comprises 49 features, such as source and destination IP addresses, port numbers, protocol types, and flow-based features. 

The UNSW-NB15 dataset is relevant to evaluating systems for encrypted data for several reasons. First, its diverse and realistic network traffic makes it suitable for evaluating systems' performance on encrypted data, as encryption is becoming increasingly prevalent in modern network environments. Second, given the variety of attack types present in the dataset, it offers a rigorous benchmark to assess the effectiveness of anomaly detection systems in identifying threats within encrypted network traffic. Third, since the dataset contains a rich set of features, it can provide insights into which features are most informative for detecting anomalies in encrypted data. By focusing on features that are still observable in encrypted traffic (e.g., packet sizes, inter-arrival times, flow duration), researchers can develop models specifically tailored to the challenges of encrypted data analysis. Finally, the dataset has been widely used in the cybersecurity research community, providing a valuable point of comparison for assessing the performance of new anomaly detection techniques against existing state-of-the-art methods in the context of encrypted data. 

To use the UNSW-NB15 dataset for encrypted data analysis, pre-processing steps include data cleaning, handling missing values, and ensuring data consistency and integrity. Feature extraction and selection for encrypted data should focus on selecting features that are still observable in encrypted network traffic, such as packet sizes, inter-arrival times, and flow duration. Utilizing domain knowledge and expert input to identify the most informative features for encrypted data analysis and employing feature selection techniques, such as Recursive Feature Elimination (RFE) or Principal Component Analysis (PCA), can help reduce dimensionality and minimize overfitting. Labeling normal and anomalous traffic patterns can be achieved by using the provided labels in the dataset to train supervised learning models or leveraging unsupervised or semi-supervised learning techniques to identify and label potential anomalies when focusing on encrypted data features. 

For data partitioning, the dataset should be divided into separate subsets for training (e.g., 70%), validation (e.g., 15%), and testing (e.g., 15%), ensuring a representative distribution of normal and anomalous traffic patterns across all subsets. If necessary, balancing class distribution can be achieved by employing resampling techniques, such as oversampling the minority class or undersampling the majority class, or utilizing synthetic data generation methods, such as SMOTE, to create additional samples of the minority class. Cost-sensitive learning approaches can also be applied to account for imbalanced class distribution during model training and evaluation. 

## Methodology 

In this experiment, we first established a baseline using traditional machine learning algorithms on the UNSW_NB15_training-set.csv dataset. The dataset was loaded into a pandas DataFrame, and categorical features ('proto', 'service', and 'state') were encoded into numerical values using LabelEncoder. The numerical features were standardized using StandardScaler to ensure consistent scaling. Unnecessary features, 'id' and 'attack_cat', were removed from the dataset. 

To identify the most important features for predicting the target variable 'label', the SelectKBest method and the mutual_info_classif function were employed. The top 10 features were selected based on their mutual information scores, and a bar plot was created to visualize the feature importance. 

The dataset was partitioned into training and testing sets, allocating 80% for training and 20% for testing, using the train_test_split function with a random state of 42 to ensure reproducibility. Four classifiers—Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM)—were defined and evaluated. Each classifier was trained on the selected features from the training set and tested on the corresponding test set. The performance of each classifier was assessed using the classification report, providing metrics such as precision, recall, F1-score, and accuracy. Confusion matrices were plotted for each classifier to visualize their performance in classifying instances into the correct categories. 

The selected features from the SelectKBest method were displayed to provide insight into the most important features for the classification task. With this information, we then turned our attention to the ThatDot Novelty Detector. The dataset was preprocessed for use in ThatDot by discretizing the data into bins using an optimizer function, which determined the optimal number of bins for each column based on the Freedman-Diaconis rule. This step transformed the continuous numerical features into categorical data, which is more suitable for the ThatDot Novelty Detector. 

Once the data was discretized, it was fed into the ThatDot Novelty Detector, and the results were extracted in JSON format. The output contained information about the novelty scores, such as probability, uniqueness, and infoContent, for each instance in the dataset. The features and target variables were extracted from the JSON output and prepared for further analysis. 

A decision tree classifier was trained on the novelty scores obtained from ThatDot to create a threshold for classification. By training the decision tree classifier, we aimed to determine an optimal boundary between the normal and anomalous instances, as identified by ThatDot. The dataset was split into training and testing sets, with 80% allocated for training and 20% for testing. The decision tree classifier was evaluated on the test set, and its performance was assessed using the classification report and accuracy score. 

Finally, the performance of the ThatDot Novelty Detector, in conjunction with the decision tree classifier, was compared to the established baseline using traditional machine learning algorithms. This comparison provided insights into the effectiveness of the ThatDot Novelty Detector in the given context and its potential for detecting novel and anomalous instances in the cybersecurity domain. 

## Results 

The experiment aimed to identify the best features for predicting malicious behavior between two endpoints in a network using the UNSW-NB15 dataset. Four machine learning classifiers were applied: Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM). 

The top 10 selected features, in no order, were found to be: 

    dur: Duration of the network connection 

    spkts: Source to destination packet count 

    dpkts: Destination to source packet count 

    sbytes: Source to destination byte count 

    dbytes: Destination to source byte count 

    rate: Transmission rate (packets/second) 

    sttl: Source to destination time-to-live value 

    dttl: Destination to source time-to-live value 

    sload: Source bits per second 

    dload: Destination bits per second 

Upon analyzing the selected features, it is observed that the extracted information is mostly from the packet headers, such as packet count, byte count, traffic rate, and TTL values. This observation is crucial because, in encrypted network environments, the payload of network packets is encrypted, making deep packet inspection techniques ineffective for detecting malicious activity. However, the features identified in this experiment can still be extracted from the packet headers, which are not encrypted. By focusing on these features, the machine learning classifiers can effectively detect intrusions, even when the network traffic payload is encrypted. 

These features provide valuable information about network traffic between two endpoints, such as duration, packet size, traffic rate, and time-to-live (TTL) values. The classifiers' performance metrics, based on these selected features, are summarized below: 

    Logistic Regression: 

         Precision: 0.84 

         Recall: 0.75 

         F1-score: 0.79 

         Accuracy: 0.83 

    Decision Tree: 

         Precision: 0.92 

         Recall: 0.92 

         F1-score: 0.92 

         Accuracy: 0.92 

    Random Forest: 

         Precision: 0.95 

         Recall: 0.94 

         F1-score: 0.94 

         Accuracy: 0.95 

    Support Vector Machine (SVM): 

         Precision: 0.87 

         Recall: 0.81 

         F1-score: 0.84 

         Accuracy: 0.87 

    ThatDot Novelty Detector with Decision Tree Classification 

         Precision: 0.85 

         Recall: 0.87 

         F1-score: 0.86 

         Accuracy: 0.87 

In this study, we compared the performance of four classifiers (logistic regression, decision tree, random forest, and support vector machine) and the novelty detector (thatDot novelty detector with decision tree classification) on the UNSW_NB15 dataset. We used the training set and testing set that were provided by the dataset authors, which contain 175,341 and 82,332 records, respectively. We evaluated the classifiers and the novelty detector based on four metrics: precision, recall, F1-score, and accuracy. 

The results show that the random forest classifier performed better than the other classifiers and the novelty detector on all four metrics. The random forest classifier achieved a precision of 0.95, a recall of 0.94, an F1-score of 0.94, and an accuracy of 0.95. The novelty detector achieved a precision of 0.85, a recall of 0.87, an F1-score of 0.86, and an accuracy of 0.87. 

One probable reason for the superior performance of the random forest classifier is that it can handle the high-dimensional and imbalanced nature of the UNSWNB15 dataset better than the other classifiers. The random forest classifier can also capture the complex interactions among the features and reduce the variance by averaging multiple decision trees. 

One likely reason for the inferior performance of the novelty detector is that it may need more data without any attack attempts to form a baseline for what is normal. It appears as if, since there are attack values all throughout the data, that attacks are being built into the system's baseline, which is a problem for analysis of the tool's performance. The novelty detector may also suffer from high false positive rates due to the diversity and similarity of the attacks in the UNSW_NB15 dataset. 

Therefore, we conclude that the random forest classifier is more suitable for detecting network intrusions in the UNSWNB15 dataset than the other classifiers and the novelty detector. However, further experiments are needed to validate these findings and to explore other methods for improving the performance of the novelty detector. 

## Conclusion 

In conclusion, this technical memo investigated the feasibility of applying machine learning techniques for anomaly detection in encrypted network data using Zeek logs. We compared the effectiveness of various machine learning approaches with the alternative commercial solution, ThatDot Novelty Detector, on the UNSW-NB15 dataset, a comprehensive network traffic dataset designed to evaluate intrusion detection systems (IDS) and other network security solutions. 

The experiment aimed to identify the best features for predicting malicious behavior between two endpoints in a network and to evaluate the performance of traditional machine learning models alongside the commercial-grade tool. Four machine learning classifiers (Logistic Regression, Decision Tree, Random Forest, and SVM) and the ThatDot Novelty Detector were compared based on their precision, recall, F1-score, and accuracy. 

The results demonstrated that the Random Forest classifier outperformed the other classifiers and the ThatDot Novelty Detector, achieving the highest scores for all performance metrics. The superior performance of the Random Forest classifier can be attributed to its ability to handle high-dimensional and imbalanced data and capture complex interactions among features. 

One important observation regarding the ThatDot Novelty Detector is its requirement for a set of data representing normal traffic to form a baseline. In the current experiment design, the Novelty Detector was being fed data that already contained attack values. Consequently, some of these attack values were inadvertently incorporated into the system's baseline, which is used to measure novelty and detect anomalies. This issue might have led to the lower performance of the Novelty Detector when compared to the Random Forest classifier. 

Although the ThatDot Novelty Detector did not perform as well as the Random Forest classifier, it still provided valuable insights into its potential use for detecting novel and anomalous instances in the cybersecurity domain. Further experiments are needed to address the issue of attack values being built into the baseline and explore other methods for improving the performance of the novelty detector. 

The selected features, mostly extracted from packet headers, proved to be valuable in detecting intrusions even when the network traffic payload is encrypted. Overall, the findings of this study highlight the potential of machine learning techniques, particularly the Random Forest classifier, for detecting network intrusions in encrypted environments. 

### Future Work 

This study has demonstrated the potential of machine learning techniques in detecting network intrusions in encrypted environments using the UNSW-NB15 dataset. However, there are several avenues for future research to further improve and validate these findings: 

    Experiment with additional machine learning algorithms: While the Random Forest classifier showed promising results, other advanced machine learning algorithms, such as deep learning, ensemble methods, or unsupervised techniques, could be explored to enhance detection performance. 

    Refine the ThatDot Novelty Detector implementation: To better evaluate the Novelty Detector, a separate set of data representing normal traffic should be provided to establish a more accurate baseline. This would help prevent the incorporation of attack values into the system's baseline and potentially improve its performance. 

    Feature engineering and selection: Investigate advanced feature engineering techniques to create new features or combine existing ones, which could lead to improved classification performance. Additionally, alternative feature selection methods could be explored to identify the most important features for intrusion detection. 

    Address class imbalance: The UNSW-NB15 dataset is highly imbalanced, with some attack types represented in significantly lower numbers than others. Future work should explore techniques to address class imbalance, such as resampling, cost-sensitive learning, or synthetic data generation, to improve the classifiers' performance. 

    Evaluate model performance on different datasets: To validate the generalizability of the findings, it is essential to test the classifiers on other network traffic datasets, such as the more recent CSE-CIC-IDS2018 dataset or real-world network data from different organizations. 

    Investigate the impact of data preprocessing: Different data preprocessing techniques, such as feature scaling, outlier removal, or data transformation, could be explored to assess their impact on the classifiers' performance. 

    Real-time intrusion detection: Evaluate the feasibility of applying the developed machine learning models for real-time intrusion detection in production environments, considering factors such as computational resources, latency, and scalability. 

    Adversarial machine learning: Cybersecurity is an adversarial domain, and attackers may attempt to evade detection systems by exploiting their weaknesses. Future work should investigate the robustness of the classifiers against adversarial attacks and develop methods to make the classifiers more resilient. 

By addressing these future research directions, we can refine and strengthen the application of machine learning techniques for network intrusion detection, particularly in encrypted network environments. This would ultimately contribute to the development of more effective and robust cybersecurity solutions. 

 

## APPENDIX 

### CONOPS (Concept of Operations) for a Data Analysis Pipeline with Zeek and Random Forest Classifier 

 

#### Objective 

The objective of this data analysis pipeline is to detect network intrusions in real-time using Zeek for data collection and a pre-trained Random Forest classifier for classification. 

#### System Overview 

The system will consist of the following components: 

    Zeek Network Analysis Framework: Collects network traffic data and extracts the top 10 features identified in the report. 

    Feature Preprocessing Module: Processes the extracted features to match the format used in the pre-trained Random Forest classifier. 

    Random Forest Classifier: A pre-trained classifier that predicts whether the network traffic is normal or malicious based on the extracted features. 

    Alerting and Reporting Module: Generates alerts and reports for potential intrusions detected by the classifier. 

#### Operational Workflow 

Data Collection: 

    Deploy Zeek on a network monitoring point, such as a network tap or SPAN port. 

    Configure Zeek with a custom script (top_features.zeek) to collect the top 10 features identified in the report and store them in a log file. 

Feature Preprocessing: 

    Continuously read the log file generated by Zeek and preprocess the extracted features. 

    Apply the same preprocessing steps used in the initial data analysis, such as encoding of categorical features and standardization of numerical features. 

Classification: 

    Feed the preprocessed features into the pre-trained Random Forest classifier. 

    The classifier will predict whether the network traffic is normal or malicious based on the input features. 

Alerting and Reporting: 

    If the classifier predicts malicious network traffic, generate an alert and notify the security operations center (SOC) or network administrators. 

    Generate reports and statistics on the detected intrusions, false positives, and system performance. 

Continuous Improvement: 

    Periodically retrain the Random Forest classifier with new and updated network traffic data to maintain high detection accuracy and adapt to new attack patterns. 

    Refine the Zeek script and feature extraction process to improve the quality and relevance of collected data. 

Considerations 

    Performance: Ensure the pipeline components can process network traffic data in real-time without causing bottlenecks or affecting network performance. 

    Security: Protect the data analysis pipeline and its components from unauthorized access and potential attacks. 

    Privacy: Ensure compliance with data protection regulations and maintain the privacy of encrypted communications. 

    Scalability: Design the pipeline to handle increasing network traffic volume and complexity, as well as the potential integration of additional data sources and classifiers. 

 

 

 
