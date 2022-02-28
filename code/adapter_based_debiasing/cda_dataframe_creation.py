import re
from utils.target_terms import *
import random
import nltk
import itertools
from itertools import *
from nltk import tokenize
import pandas as pd
from tqdm import *
import argparse
from datasets import load_dataset

nltk.download('punkt')


# Find all target terms in the original sentence that have to be replaced
def findTargetTerms(phrase, target_terms):
    found_terms = []
    for entry in target_terms:
        if string_found(entry, phrase):
            found_terms.append(entry)
    return found_terms


# Find all opposite target terms for each found term
def findOppositeTargetTerm(tt_list, target_term_frame):
    tt_opp_list = []
    for entry in tt_list:
        if entry in target_term_frame['T1'].tolist():
            aggregatedFrame = target_term_frame.groupby(['T1']).agg(T2=('T2', list)).reset_index()
            tt_opp_list.append(aggregatedFrame.loc[aggregatedFrame['T1'] == entry, 'T2'].iloc[0])
        elif entry in target_term_frame['T2'].tolist():
            aggregatedFrame = target_term_frame.groupby(['T2']).agg(T1=('T1', list)).reset_index()
            tt_opp_list.append(aggregatedFrame.loc[aggregatedFrame['T2'] == entry, 'T1'].iloc[0])
    return tt_opp_list


# Create all combinations of target terms and opposite target terms
def createCombination(tt_list, tt_opp_list):
    combiList = []
    for t1 in range(len(tt_list)):
        combiList.append((tt_list[t1], random.choice(tt_opp_list[t1])))
    return combiList


# Create list of opposing sentences where all target terms are replaced in all possible combinations
def replaceTermsInSentence(biased_sentence, tt_combi):
    tmp_sentence = biased_sentence
    rep_dict = {}
    for tup in tt_combi:
        rep_dict[tup[0]] = tup[1]
    regex = re.compile("|".join([r"\b{}\b".format(t) for t in rep_dict]))
    opp_sentence = regex.sub(lambda m: rep_dict[m.group(0)], tmp_sentence.rstrip())
    return opp_sentence


def string_found(string1, string2):
    if re.search(r"\b" + re.escape(string1) + r"\b", string2):
        return True
    return False

# Create Cartesian product
def cartesianProductList(list1, list2):
    cartesianProductList = []
    for element in itertools.product(list1, list2):
        cartesianProductList.append(element)
    return cartesianProductList


def preprocessComment(comment):
    # Remove line breaks
    comment = re.sub(r'\n\s*\n', '\n', comment)
    comment = comment.replace("\n", ".")
    # Create whitespace after punctuation
    comment = re.sub(r'\.(\w)', r'. \1', comment)
    # split block to list of sentences
    s_list = tokenize.sent_tokenize(comment)
    return s_list

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="The output for the counterfactual augmented dataset.")
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="The input file containing the original data set to apply cda to.")
    parser.add_argument("--bias_type",
                        type=str,
                        required=True,
                        help="For which bias type should CDA be applied? Options are: 'religious', 'sexual' or 'both'")
    parser.add_argument("--sentence_selection_strategy",
                        type=str,
                        required=True,
                        help="What strategy should be used to create the data set? Options are: 'target_attribute_combi' or 'all_targets'")
    parser.add_argument("--cda_strategy",
                        type=str,
                        required=True,
                        help="What CDA strategy should be used? Options are: 'one_sided' or 'two_sided'")
    parser.add_argument("--replacement_strategy",
                        type=str,
                        required=True,
                        help="What replacement strategy should be used? Options are: 'one_way' or 'two_ways'")
    parser.add_argument("--dataset_size",
                        type=str,
                        required=True,
                        help="Should the dataset include the original unbiased sentences or only those containing the target word? Options are: 'total' or 'subset'")
    parser.add_argument("--cache_dir",
                        type=str,
                        required=True,
                        help="Directory to store the dataset cache")
    parser.add_argument("--preprocessing_num_workers",
                        type=int,
                        required=True,
                        help="How many workers should be used for preprocessing the dataset?")
    args = parser.parse_args()

    print('CDA Dataframe is going to be created for the following Settings:')
    print('Sentence Selection Strategy: ' + str(args.sentence_selection_strategy))
    print('CDA Strategy: ' + str(args.cda_strategy))
    print('Bias Type: ' + str(args.bias_type))
    print('Replacement Strategy: ' + str(args.replacement_strategy))

    cda_sentences = []


    tt_dir_sb = '../data/target_term_pairs/Target_Term_Pairs_Sexual_Orientation.csv'
    tt_dir_rb = '../data/target_term_pairs/Target_Term_Pairs_Religious_Bias.csv'

    if args.bias_type == 'religious':
        # Retrieve cartesian product of target terms
        t1, t2, a1, a2 = religious_bias_tt()
        t1 = [x.lower() for x in t1]
        a1 = [x.lower() for x in a1]

        T1_A1_cartesianProduct = cartesianProductList(t1, a1)

        # Retrieve Target Term Pairs
        tt_pairs = pd.read_csv(tt_dir_rb)

    elif args.bias_type == 'sexual':
        # Retrieve cartesian product of target terms
        t1, t2, a1, a2 = sexual_bias_tt()
        t1 = [x.lower() for x in t1]
        a1 = [x.lower() for x in a1]

        T1_A1_cartesianProduct = cartesianProductList(t1, a1)

        # Retrieve Target Term Pairs
        tt_pairs = pd.read_csv(tt_dir_sb)


    elif args.bias_type == 'both':
        print('Both bias types should be calculated')

        t1_sb, t2_sb, a1_sb, a2_sb = sexual_bias_tt()
        t1_sb = [x.lower() for x in t1_sb]
        a1_sb = [x.lower() for x in a1_sb]

        T1_A1_sb = cartesianProductList(t1_sb, a1_sb)

        t1_rb, t2_rb, a1_rb, a2_rb = religious_bias_tt()
        t1_rb = [x.lower() for x in t1_rb]
        a1_rb = [x.lower() for x in a1_rb]

        T1_A1_rb = cartesianProductList(t1_rb, a1_rb)

        T1_A1_cartesianProduct = T1_A1_sb + T1_A1_rb

        # Retrieve Target Term Pairs
        tt_pairs_sb = pd.read_csv(tt_dir_sb)
        tt_pairs_rb = pd.read_csv(tt_dir_rb)

        tt_pairs = pd.concat([tt_pairs_sb, tt_pairs_rb])




    dataset = load_dataset("text", data_files={"train": args.input_file}, cache_dir=args.cache_dir)

    dataset = dataset['train']

    preprocessed_dataset = dataset.map(
        lambda example: {'sentences': preprocessComment(example['text'])},
        num_proc=args.preprocessing_num_workers
    )

    s_list = preprocessed_dataset['sentences']

    flat_s_list = []
    for sublist in s_list:
        for item in sublist:
            flat_s_list.append(item)

    print('List of sentences created.')



    # Swap sentences
    # Create list of target terms
    if args.replacement_strategy == 'one_way':
        tt_list = list(set(tt_pairs['T1'].tolist()))
    elif args.replacement_strategy == 'two_ways':
        tt_list = list(set(tt_pairs['T1'].tolist())) + list(set(tt_pairs['T2'].tolist()))
    else:
        print('The replacement strategy chosen does not exist! Please choose between "one_way" or "two_ways"!')

    # Create counter to check progress
    sentence_counter = 0
    biased_sentences_counter = 0

    if (args.sentence_selection_strategy == 'target_attribute_combi'):
        for sentence in flat_s_list:
            bias_found = False

            sentence = sentence.strip().lower()
            if len(sentence.split()) < 5:
                continue

            if (sentence_counter % 100000) == 0:  # status update
                print("Processed: ", sentence_counter, " / ", len(flat_s_list), " Sentences")

            # As many sentences in wikipedia start with "category:" we remove this term in the beginning of sentences
            sentence = re.sub(r'^category:[^\S]*', '', sentence)

            for tup in T1_A1_cartesianProduct:
                if tup[0] in sentence and tup[1] in sentence:
                    if string_found(tup[0], sentence) and string_found(tup[1], sentence):
                        biased_sentence = sentence
                        found_tt = findTargetTerms(sentence, tt_list)
                        tt_opp_list = findOppositeTargetTerm(found_tt, tt_pairs)
                        tt_combi = createCombination(found_tt, tt_opp_list)
                        opposing_sentence = replaceTermsInSentence(biased_sentence, tt_combi)
                        if args.cda_strategy == 'one_sided':
                            cda_sentences.append(opposing_sentence)
                        if args.cda_strategy == 'two_sided':
                            cda_sentences.append(biased_sentence)
                            cda_sentences.append(opposing_sentence)
                        biased_sentences_counter+=1
                        bias_found = True
                        break

            if not bias_found and args.dataset_size == 'total':
                cda_sentences.append(sentence)
            sentence_counter += 1


    elif (args.sentence_selection_strategy == 'all_targets'):
        for sentence in flat_s_list:
            bias_found = False

            sentence = sentence.strip().lower()
            if len(sentence.split()) < 5:
                continue

            if (sentence_counter % 100000) == 0:  # status update
                print("Processed: ", sentence_counter, " / ", len(flat_s_list), " Sentences")

            # As many sentences in wikipedia start with "category:" we remove this term in the beginning of sentences
            sentence = re.sub(r'^category:[^\S]*', '', sentence)

            for target in tt_list:
                if target in sentence:
                    if string_found(target, sentence):
                        biased_sentence = sentence
                        found_tt = findTargetTerms(sentence, tt_list)
                        tt_opp_list = findOppositeTargetTerm(found_tt, tt_pairs)
                        tt_combi = createCombination(found_tt, tt_opp_list)
                        opposing_sentence = replaceTermsInSentence(biased_sentence, tt_combi)
                        if args.cda_strategy == 'one_sided':
                            cda_sentences.append(opposing_sentence)
                        if args.cda_strategy == 'two_sided':
                            cda_sentences.append(biased_sentence)
                            cda_sentences.append(opposing_sentence)
                        bias_found = True
                        biased_sentences_counter += 1
                        break

            if not bias_found and args.dataset_size == 'total':
                cda_sentences.append(sentence)
            sentence_counter += 1


    print(str(biased_sentences_counter) + " Biased Sentences found for Target Term Category")
    print('Length of CDA dataset: ' + str(len(cda_sentences)))


    # Write to file for reconstructability
    with open(args.output_file, 'w', encoding='utf8') as training_file:
        for arg in cda_sentences:
            training_file.write(arg)
            training_file.write('\n')

if __name__ == "__main__":
    main()