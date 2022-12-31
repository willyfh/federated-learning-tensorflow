# Reproducing Federated Learning
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ooufYAIuKVGVozLOR_WoQjjCmgAab2_W?usp=sharing)

This is an unofficial implementation of federated learning (FedAvg). I reproduced some experiment results in the federated learning paper around the beginning of 2021. In this repository, I provided the implementation along with the reproduced results.

**Paper**: McMahan, H. B. et al. “Communication-Efficient Learning of Deep Networks from Decentralized Data.” International Conference on Artificial Intelligence and Statistics (2017) > https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf


## Summary of the paper
I summarize the paper in the following slides:

https://github.com/willyfh/federated-learning/blob/main/doc/Federated%20Learning%20-%20Summary.pdf

## Requirements
- matplotlib==3.3.4
- tensorflow==2.4.1
- keras==2.4.3
- scikit-learn==0.24.1
- tqdm==4.57.0

## Python Files (.py) 
The python files (federated_train.py & plot_multi_cases.py) are provided for a simpler execution.
Basically the implementation is the same as the provided Jupyter Notebook file (which was actually used to obtained the results shown in this project).

1. Install the required libraries:
> pip install -r requirements.txt

2. Execute federated_train.py to train the model:
Example for the case of 2NN (MLP), Non-IID, C=0, E=1, B=all, Learning rates={0.1, 0.01}, target_accuracy=0.93, communication_rounds=200
> python federated_train.py --model mlp --data_dist noniid --n_clients 100 --c 0 --e 1 --b -1 --c_rounds 200 --lr 0.1 0.01 --target_acc 0.93

3. After the execution is completed, the training result file (.pickle) and the plots (train-test plot) will be stored in the same directory as the python file.

Arguments info:
```
  --model {mlp,cnn}        mlp or cnn
  --data_dist {iid,noniid} iid or noniid
  --n_clients              number of clients
  --c                      client fraction [0.0,1.0]
  --e                      client epoch
  --b                      batch size. input -1 to use a single batch
  --c_rounds               communication rounds
  --lr                     learning rate. separate by space for multiple learning rates. eg. --lr 0.1 0.001
  --target_acc             target test accuracy [0.0, 1.0]
```

For generating multi cases plot, we need to have all of the training result files (for all cases) which were created using the above steps.
Example for CNN IID case:
> python plot_multi_cases.py --model cnn --data_dist iid --result_files train_result_all_0.1_1_cnn_iid.pickle train_result_50_0.1_1_cnn_iid.pickle train_result_10_0.1_1_cnn_iid.pickle train_result_all_0.1_20_cnn_iid.pickle train_result_50_0.1_20_cnn_iid.pickle train_result_10_0.1_20_cnn_iid.pickle

Arguments info:
```
  --model {mlp,cnn}        mlp or cnn
  --data_dist {iid,noniid} iid or noniid
  --result_files           training result files. separate by space for multiple files. eg. --result_files file1.pickle file2.pickle
```

## Jupyter Notebook File (.ipynb)
The results shown in this project were obtained using the provided Jupyter Notebook file (federated_learning.ipynb).
It was executed and tested in the Google Colab environment.

1. Upload federated_learning.ipynb to Google Colab / Jupyter Notebook
2. Change the parameters as needed in the provided section "Parameters".
3. Click Runtime > Run all
4. The training result file (.pickle) will be saved and the plots (train-test plot) will be displayed in the notebook.

For generating multi cases plot, we need to upload all of the training result files (for all cases) which were created using the above steps
then execute the last section of the notebook (Generate Multi Cases Plots) to generate the plots.

## Reproduced Results

### Reproducing Table 1 in the paper for C = 0.0, 0.1, 1.0.
![image](https://user-images.githubusercontent.com/5786636/209909399-7a99b2bb-4fba-431b-9b2f-594d37bb466a.png)

Due to limited computational power, I ran only 200-500 rounds with a single learning rate for producing the following results. Consequently, I adjusted the target-test accuracy to become 93% and 97% for 2NN and CNN, respectively.

Generally, we can see similar results to the paper. With B=∞, there is only a small advantage in increasing C. Using smaller B=10 shows a significant improvement in using C >= 0.1, especially in the non-IID case.

*Please see the appendix below for accuracy and loss plot for each case in the table
above.*

### Reproducing Figure 2 in the paper for MNIST CNN with (B, E) = (10,1), (10, 20), (50,1), (50, 20), (∞,1), (∞,20)
<p float="left">
  <img src="https://user-images.githubusercontent.com/5786636/209910408-eb33f7d3-9644-4740-b067-7446266bd452.png" width="49%" />
  <img src="https://user-images.githubusercontent.com/5786636/209910456-0ebed79e-10fe-447a-8af4-bb1de2dc48d3.png" width="49%" /> 
</p>

Here, I ran only 200 rounds with a single learning rate for this case due to limited computational power.

Generally, we also can see similar results to the paper. With C=0.1, adding more local updates per round (increase E & decrease B) can produce a significant decrease in communication costs.



## Appendix
### 2NN, IID, E=1, B=∞, C=0
![image](https://user-images.githubusercontent.com/5786636/209913681-78f5f903-3448-45b4-96ed-78e6f9010f63.png)

### 2NN, IID, E=1, B=∞, C=0.1
![image](https://user-images.githubusercontent.com/5786636/209913718-caa4b6a9-dbd4-4f08-8fee-784b97f5041a.png)

### 2NN, IID, E=1, B=∞, C=1
![image](https://user-images.githubusercontent.com/5786636/209913763-5389974b-c81f-4d0d-8c7d-6d9ea4e9a0c2.png)

### 2NN, IID, E=1, B=10, C=0
![image](https://user-images.githubusercontent.com/5786636/209912610-cd949812-437e-4c1e-8c12-bd6e09241fd3.png)

### 2NN, IID, E=1, B=10, C=0.1
![image](https://user-images.githubusercontent.com/5786636/209912648-0bb6b8ac-5c08-459c-8f8c-9ae9cd0e58f4.png)

### 2NN, IID, E=1, B=10, C=1
![image](https://user-images.githubusercontent.com/5786636/209912869-06825fa9-c0c4-4f0a-92e1-3cb8fc5239a4.png)

### 2NN, Non-IID, E=1, B=∞, C=0
![image](https://user-images.githubusercontent.com/5786636/209912917-cca2b3c6-a7a8-4226-b734-ca6859083c7d.png)

### 2NN, Non-IID, E=1, B=∞, C=0.1
![image](https://user-images.githubusercontent.com/5786636/209912973-88316428-deb2-474e-8c4c-e2588433deea.png)

### 2NN, Non-IID, E=1, B=∞, C=1
![image](https://user-images.githubusercontent.com/5786636/209913014-d47d3d66-45a5-458b-92db-ffaaccb6df43.png)

### 2NN, Non-IID, E=1, B=10, C=0
![image](https://user-images.githubusercontent.com/5786636/209913049-1975c73d-2946-4071-9c21-950f6ef7df40.png)

### 2NN, Non-IID, E=1, B=10, C=0.1
![image](https://user-images.githubusercontent.com/5786636/209913092-182c5df1-a5fa-4b5f-baf5-6332cc469d20.png)

### 2NN, Non-IID, E=1, B=10, C=1
![image](https://user-images.githubusercontent.com/5786636/209913120-e3b94dae-4e8e-45fd-aa5b-c55bbf91e331.png)

### CNN, IID, E=5, B=∞, C=0
![image](https://user-images.githubusercontent.com/5786636/209913153-677210e7-7ec3-41b4-b279-c3b1188227d4.png)

### CNN, IID, E=5, B=∞, C=0.1
![image](https://user-images.githubusercontent.com/5786636/209913201-89483a86-191e-4c7f-97ad-def49430463a.png)

### CNN, IID, E=5, B=∞, C=1
![image](https://user-images.githubusercontent.com/5786636/209913235-201a2e8a-b40f-4ce3-aaeb-fd6f1fa0d4c9.png)

### CNN, IID, E=5, B=10, C=0
![image](https://user-images.githubusercontent.com/5786636/209913261-44129819-f1cb-4bc9-adb7-da402e7ad26f.png)

### CNN, IID, E=5, B=10, C=0.1
![image](https://user-images.githubusercontent.com/5786636/209913305-4e1119c7-0456-4016-8778-f2d8a8b194dd.png)

### CNN, IID, E=5, B=10, C=1
![image](https://user-images.githubusercontent.com/5786636/209913334-26c0d718-bde5-4bae-8ee0-59aa21f70e2d.png)

### CNN, Non-IID, E=5, B=∞, C=0
![image](https://user-images.githubusercontent.com/5786636/209913383-5d52525e-7188-4fd8-85e8-343d92ac96d3.png)

### CNN, Non-IID, E=5, B=∞, C=0.1
![image](https://user-images.githubusercontent.com/5786636/209913410-ca63576f-cb75-499b-b68d-c5da03fea1bd.png)

### CNN, Non-IID, E=5, B=∞, C=1
![image](https://user-images.githubusercontent.com/5786636/209913458-0080ab53-bf70-4260-b5b9-67a16c3f86b4.png)

### CNN, Non-IID, E=5, B=10, C=0
![image](https://user-images.githubusercontent.com/5786636/209913543-2e31640b-3815-4fba-a8f8-28696c6b98dc.png)

### CNN, Non-IID, E=5, B=10, C=0.1
![image](https://user-images.githubusercontent.com/5786636/209913590-61737c2b-1b87-41b7-a5e3-0a85899ab155.png)

### CNN, Non-IID, E=5, B=10, C=1
![image](https://user-images.githubusercontent.com/5786636/209913610-31cdab99-c615-498d-9e07-762e8d43347c.png)
