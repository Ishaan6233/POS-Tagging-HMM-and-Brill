# POS-Tagging-and-Evaluation

## **Installation & Execution**

### **Installation**
Before running the script, install the required dependencies:
```
pip install nltk pandas
```

### **Running the Script**
The script requires the following command-line arguments:
- `--tagger`: Choose the tagger type (`hmm` or `brill`).
- `--train`: Path to the training corpus file.
- `--test`: Path to the testing corpus file.
- `--output`: Path to save the output file.

#### **Example Usage:**
Run the **HMM tagger** with training and testing data:
```
python3 src/main.py --tagger hmm --train data/train.txt --test data/test.txt --output output/test_hmm.txt
```
Run the **Brill tagger** with the same data:
```
python3 src/main.py --tagger brill --train data/train.txt --test data/test.txt --output output/test_brill.txt
```

## **Data Description**

- **Training Data**: Found in [`data/train.txt`](data/train.txt), this file consists of sentences where each word is tagged with its corresponding POS tag.
- **In-Domain Test Data**: Located in [`data/test.txt`](data/test.txt), this dataset is used to evaluate the taggers on the same domain as training data.
- **Out-of-Domain Test Data**: Available in [`data/test_ood.txt`](data/test_ood.txt), this dataset tests how well the trained models generalize to unseen domains.

Ensure that all data files follow the required format, with each word-tag pair separated by a space and sentences separated by blank lines.

