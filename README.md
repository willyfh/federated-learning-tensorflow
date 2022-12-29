# Reproducing Federated Learning
**Paper**: McMahan, H. B. et al. “Communication-Efficient Learning of Deep Networks from Decentralized Data.” International Conference on Artificial Intelligence and Statistics (2016). **[PDF]**: https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
## Requirements
- matplotlib==3.3.4
- tensorflow==2.4.1
- keras==2.4.3
- scikit-learn==0.24.1
- tqdm==4.57.0

## Explanations
Here, I provide two versions of the implementation:
1) Python file (.py)
2) Jupyter Notebok file (.py)

Please see the below explanation for how to run the codes.

### Python Files (.py) 
The python codes (federated_train.py & plot_multi_cases.py) are provided for a simpler execution.
Basically the implementation is the same as the provided Jupyter Notebook file (which was actually used to obtained the results shown in the report).

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

### Jupyter Notebook File (.ipynb)
The results shown in the report were obtained using the provided Jupyter Notebook file (federated_learning.ipynb).
It was executed and tested in the Google Colab environment.

1. Upload federated_learning.ipynb to Google Colab / Jupyter Notebook
2. Change the parameters as needed in the provided section "Parameters".
3. Click Runtime > Run all
4. The training result file (.pickle) will be saved and the plots (train-test plot) will be displayed in the notebook.

For generating multi cases plot, we need to upload all of the training result files (for all cases) which were created using the above steps
then execute the last section of the notebook (Generate Multi Cases Plots) to generate the plots.

