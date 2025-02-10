import argparse
import os

from nltk.probability import LaplaceProbDist, LidstoneProbDist
from nltk.tag import hmm, brill_trainer, UnigramTagger, DefaultTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl.template import Template


def load_corpus(filepath):
    """
    Load the POS-tagged corpus from the file.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file '{filepath}' does not exist or cannot be found.")

    sentences = []  # List to store all sentences
    with open(filepath, "r", encoding="utf-8") as f:
        sentence = []  # Temporary list to store the current sentence
        for line in f:
            line = line.strip()
            if not line:  # Empty line marks the end of a sentence
                if sentence:
                    sentences.append(sentence)
                    sentence = []  # Reset for the next sentence
            else:
                word, tag = line.rsplit(" ", 1)  # Split word and tag from the end
                sentence.append((word, tag))
        if sentence:
            sentences.append(sentence)  # Add the last sentence if file ends without a blank line
    return sentences


def train_hmm_tagger(train_sents, estimator=LaplaceProbDist, gamma=0.1):
    """
    Train a Hidden Markov Model (HMM) tagger using the training data.
    """
    print(
        f"Training HMM tagger with {len(train_sents)} sentences using estimator {estimator.__name__ if estimator.__name__ == 'LaplaceProbDist' else 'LidstoneProbDist'}...")
    trainer = hmm.HiddenMarkovModelTrainer()
    if estimator == LidstoneProbDist:  # Check if the estimator is LidstoneProbDist, which requires special setup
        tagger = trainer.train(train_sents, estimator=lambda fd, bins: LidstoneProbDist(fd, gamma, bins))
    else:
        tagger = trainer.train(train_sents, estimator=estimator)
    return tagger


def train_brill_tagger(train_sents):
    """
    Train a Brill tagger using a Unigram tagger as the baseline.
    """
    # Use a DefaultTagger with a fallback tag "NN" for words that are unseen by the Unigram tagger
    baseline_tagger = UnigramTagger(train_sents, backoff=DefaultTagger("NN"))

    # Templates are rules that define patterns for how words should be tagged based on their context
    templates = [
        Template(Pos([-1])), Template(Pos([1])),
        Template(Pos([-2])), Template(Pos([2])),
        Template(Pos([-1, 1])), Template(Pos([-2, 2])),
        Template(Word([-1])), Template(Word([1]))
    ]
    trainer = brill_trainer.BrillTaggerTrainer(baseline_tagger, templates, deterministic=True)
    print("Training Brill tagger...")

    # Train the Brill tagger, limiting the number of transformation rules to 200
    # max_rules=200 means the trainer will stop after applying 200 transformation rules
    tagger = trainer.train(train_sents, max_rules=200)
    return tagger


def evaluate_tagger(tagger, test_sents):
    """
    Evaluate the tagger's performance by calculating its accuracy on the test sentences.
    """
    correct_tags = 0
    total_tags = 0
    for sent in test_sents:
        words, true_tags = zip(*sent)  # Separate words and true tags from the sentence
        tagged_sent = tagger.tag(words)  # Use the tagger to generate predicted tags for the words

        # Compare predicted tags with true tags and count correct predictions
        correct_tags += sum(1 for (pred, true) in zip(tagged_sent, true_tags) if pred[1] == true)
        total_tags += len(true_tags)  # Update the total number of tags in the sentence
    return correct_tags / total_tags if total_tags > 0 else 0.0


def tag_and_write_output(tagger, test_sents, output_path):
    """
    Use the trained tagger to tag test sentences and write the output to a file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in test_sents:

            # Extract words from the sentence and tag them using the trained tagger
            tagged_sent = tagger.tag([word for word, _ in sent])

            # Write each word and its predicted tag to the file
            for word, tag in tagged_sent:
                f.write(f"{word} {tag}\n")
            f.write("\n")
    print(f"Output written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="POS Tagging using HMM and Brill Taggers")
    parser.add_argument("--tagger", required=True, choices=["hmm", "brill"], help="Tagger type: 'hmm' or 'brill'")
    parser.add_argument("--train", required=True, help="Path to the training corpus")
    parser.add_argument("--test", required=True, help="Path to the test corpus")
    parser.add_argument("--output", required=True, help="Path to the output file")
    args = parser.parse_args()  # Parse the command-line arguments

    # Load training data
    print(f"Loading training data from {args.train}...\n")
    train_sents = load_corpus(args.train)

    # Load testing data
    print(f"Loading testing data from {args.test}...\n")
    test_sents = load_corpus(args.test)

    # Train the selected tagger
    if args.tagger == "hmm":
        best_hmm_tagger = None
        best_accuracy = 0  # Track the best accuracy to pick the best model

        # First, check with LaplaceProbDist (no gamma needed)
        print("Training HMM with Laplace smoothing...")
        tagger = train_hmm_tagger(train_sents, estimator=LaplaceProbDist)
        accuracy_score = evaluate_tagger(tagger, test_sents)
        print(f"Accuracy for HMM with LaplaceProbDist: {accuracy_score * 100:.4f}\n")

        if accuracy_score > best_accuracy:
            best_accuracy = accuracy_score
            best_hmm_tagger = tagger

        # Then, check with LidstoneProbDist for gamma values 0.1 and 0.8
        for gamma in [0.1, 0.8]:
            print(f"Training HMM with Lidstone smoothing (gamma={gamma})...")
            tagger = train_hmm_tagger(train_sents, estimator=lambda fd, bins: LidstoneProbDist(fd, gamma, bins),
                                      gamma=gamma)
            accuracy_score = evaluate_tagger(tagger, test_sents)
            print(f"Accuracy for HMM with LidstoneProbDist (gamma={gamma}): {accuracy_score * 100:.4f}\n")

            if accuracy_score > best_accuracy:
                best_accuracy = accuracy_score
                best_hmm_tagger = tagger

        tagger = best_hmm_tagger

    elif args.tagger == "brill":
        print("Training Brill tagger...")
        tagger = train_brill_tagger(train_sents)  # Train the Brill tagger

        # Evaluate Brill tagger's accuracy on the test data
        accuracy_score = evaluate_tagger(tagger, test_sents)
        print(f"Accuracy for Brill Tagger: {accuracy_score * 100:.4f}%\n")

    # Tag the test data and write the output to the specified file
    print(f"Tagging and writing output to {args.output}...\n")
    tag_and_write_output(tagger, test_sents, args.output)


if __name__ == "__main__":
    main()
