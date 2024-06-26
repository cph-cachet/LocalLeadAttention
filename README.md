Link to Paper : https://ieeexplore.ieee.org/document/10364166

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reading-between-the-leads-local-lead/ecg-classification-on-physionet-challenge-1)](https://paperswithcode.com/sota/ecg-classification-on-physionet-challenge-1?p=reading-between-the-leads-local-lead)

Results on local Validation

| AUPRC | AUROC | Accuracy  | F-Measure | Challenge Score  |
| :-----: | :---: | :---: | :---: | :---: |
| 0.548 | 0.949   | 0.418   | 0.521   | 0.696   |



# Python code for the PhysioNet/Computing in Cardiology Challenge 2021

## How do I run these scripts?

You can run this classifier code by installing the requirements

    pip install requirements.txt

and running

    python train_model.py training_data model
    python test_model.py model test_data test_outputs

where `training_data` is a folder of training data files, `model` is a folder for saving your models, `test_data` is a folder of test data files (you can use the training data locally for debugging and cross-validation), and `test_outputs` is a folder for saving your models' outputs. The [PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/) provides training databases with data files and a description of the contents and structure of these files.

After training your model and obtaining test outputs with above commands, you can evaluate the scores of your models using the [PhysioNet/CinC Challenge 2021 evaluation code](https://github.com/physionetchallenges/evaluation-2021) by running

    python evaluate_model.py labels outputs scores.csv class_scores.csv

where `labels` is a folder containing files with one or more labels for each ECG recording, such as the training database on the PhysioNet webpage; `outputs` is a folder containing files with outputs produced by your models for those recordings; `scores.csv` (optional) is a collection of scores for your models; and `class_scores.csv` (optional) is a collection of per-class scores for your models.


## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies that you can run reliably in other computing environments and operating systems.

To guarantee that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a subset of the training data.

If you have trouble running your code, then please try the follow steps to run the example code, which is known to work.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data test_data model test_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2021/#data-access). Put some of the training data in `training_data` and `test_data`. You can use some of the training data to check your code (and should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/physionetchallenges/python-classifier-2021.git

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        model  python-classifier-2021  test_data  test_outputs  training_data

        user@computer:~/example$ ls training_data/
        A0001.hea  A0001.mat  A0002.hea  A0002.mat  A0003.hea  ...

        user@computer:~/example$ cd python-classifier-2021/

        user@computer:~/example/python-classifier-2021$ docker build -t image .

        Sending build context to Docker daemon  30.21kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/python-classifier-2021$ docker run -it -v ~/example/model:/physionet/model -v ~/example/test_data:/physionet/test_data -v ~/example/test_outputs:/physionet/test_outputs -v ~/example/training_data:/physionet/training_data image bash

        root@[...]:/physionet# ls
            Dockerfile             model             test_data      train_model.py
            extract_leads_wfdb.py  README.md         test_model.py
            helper_code.py         requirements.txt  test_outputs
            LICENSE                team_code.py      training_data

        root@[...]:/physionet# python train_model.py training_data model

        root@[...]:/physionet# python test_model.py model test_data test_outputs

        root@[...]:/physionet# exit
        Exit

        user@computer:~/example/python-classifier-2021$ cd ..

        user@computer:~/example$ ls test_outputs/
        A0006.csv  A0007.csv  A0008.csv  A0009.csv  A0010.csv  ...

## How do I learn more?

Please see the [PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges).

## Useful links

* [The PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/)
* [MATLAB example code for the PhysioNet/CinC Challenge 2021](https://github.com/physionetchallenges/matlab-classifier-2021)
* [Evaluation code for the PhysioNet/CinC Challenge 2021](https://github.com/physionetchallenges/evaluation-2021) 
* [2021 Challenge Frequently Asked Questions (FAQ)](https://physionetchallenges.org/2021/faq/) 
* [Frequently Asked Questions (FAQ)](https://physionetchallenges.org/faq/) 
