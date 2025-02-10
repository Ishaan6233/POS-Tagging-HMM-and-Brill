# Comparative Analysis of POS Tagging with HMM and Brill

## Project Description

- This project involves training and evaluating two part-of-speech (POS) taggers: the **Hidden Markov Model (HMM) tagger** and the **Brill tagger**, on part-of-speech tagged sentences.
- The goal is to compare the performance of these two taggers on **in-domain** and **out-of-domain** text samples from the Georgetown University Multilayer Corpus (GUM).
- With a focus on **tuning**, **comparing**, and analyzing the **accuracy** of these models for POS tagging.

---

## 3-rd Party Libraries

* `main.py L:01` used `argparse` for parsing command-line arguments and specifying paths for the input and output files.

---

## Execution
To ensure that all required libraries are installed. Prior to running the program, run the following command: `pip install nltk pandas`

To run the script, you need to provide four command-line arguments:
- **--tagger**: Choose the tagger type (hmm or brill).
- **--train**: Path to the training corpus file.
- **--test**: Path to the testing corpus file.
- **--output**: Path to save the output file.
  
Example usage:

- Running the HMM tagger on a specific training and testing set: `python3 src/main.py --tagger hmm --train data/train.txt --test data/test.txt --output output/test_hmm.txt`
- Similarly, for the Brill tagger : `python3 src/main.py --tagger brill --train data/train.txt --test data/test.txt --output output/test_brill.txt`

---

## Project Directory Structure
```
f24-asn4-ishhaan6233/
  ├─ data/                                 # Directory containing the input files (training and testing corpora)
  │  ├─ test.txt                           # Test corpus file
  │  ├─ test_odd.txt                       # Out-of-domain test corpus file
  │  ├─ train.txt                          # Training corpus file
  ├─ output/                               # Directory where output files are saved
  │  ├─ test_brill.txt                     # Output file for Brill POS tagger
  │  ├─ test_hmm.txt                       # Output file for HMM POS tagger
  │  ├─ test_ood_brill.txt                 # Output file for Brill POS tagger
  │  ├─ test_ood_hmm.txt                   # Output file for HMM POS tagger on out-of-domain data
  ├─ src/                                  # Source code directory
  │  ├─ main.py                            # Main script that implements the POS tagging models and handles the logic for training, testing, and output generation
  ├─ README.md                             # README file that provides instructions on how to set up and run the project
  ├─ report.pdf                            # Project report in PDF format
```
---

## Functions of src/main.py
- **load_data(file_path)**: Loads data from the given file, returns a list of sentences (each sentence is a list of word-tag pairs).
- **train_hmm(train_data)**: Trains an HMM POS tagger using the provided training data and applies smoothing (Laplace or Lidstone).
- **train_brill(train_data)**: Trains a Brill tagger with the given training data, refining an initial tagger using transformation rules.
- **evaluate_tagger(tagger, test_data)**: Evaluates the tagger’s accuracy on test data by comparing predicted and actual tags.
- **tag_data(tagger, test_data)**: Tags the test data using the trained tagger.
- **save_output(output_file, tagged_data)**: Saves the tagged data to the specified output file.
- **parse_arguments()**: Parses command-line arguments to configure the script.
- **main()**: Orchestrates loading data, training, evaluating, tagging, and saving results based on user input.


---

## Data

- Training Data: The training data is a .txt file where each line contains a word followed by its Part-Of-Speech (POS) tag, separated by a space. Sentences should be separated by blank lines.

- Test Data: The test data follow the same format as the training data. This file is used to evaluate the trained tagger's performance.

- Output: The output will be a .txt file with each word from the test data and the POS tag assigned by the trained model. Each sentence is separated by a blank line.


---

## Observations


| Model      |      Smoothing Method      |      In-domain Accuracy (test.txt)      |      Out-of-domain Accuracy (test_odd.txt)      |
-------------|----------------------------|-----------------------------------------|-------------------------------------------------|
| HMM        |     Laplace Smoothing	    |                 77.19%                  |                  68.90%                         |
| HMM	       |      Lidstone (gamma=0.1)  |	                82.53%                  |	                 77.29%                         |
| HMM	       |      Lidstone (gamma=0.8)  |	                77.88%                  |                  71.05%                         |
| Brill      |	No smoothing (default)    |	                 83.17%                 |                  80.90%                         |


---

## References

1. **NLTK POS Tagging Guide**: [Official NLTK documentation for implementing POS tagging with NLTK](https://www.nltk.org/book/ch05.html)
2. **Tutorial on HMM for POS Tagging**: [Explains the theory and implementation of HMM for tagging](https://www.tutorialspoint.com/natural_language_processing/natural_language_processing_hidden_markov_model.htm)
3. **NLTK Brill Tagger Documentation**: [Official NLTK guide to the Brill tagger implementation](https://www.nltk.org/_modules/nltk/tag/brill.html)
4. **Implementing LidstoneProbDist into the code**: [Official Documentation for the LidstoneProbDist since this was somehting new to me](https://www.nltk.org/api/nltk.probability.LidstoneProbDist.html)
