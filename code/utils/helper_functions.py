import itertools
import pandas as pd
import re
from scipy import stats
from tqdm import tqdm
import logging
import json_lines

def cartesianProductList(list1, list2):
    """
    Creates Cartesian Product of two term-lists.

    Parameters
    ----------
    list1 : list of str
    A list containing multiple terms.
    list2 : list of str
    A list containing multiple terms.

    Returns
    -------
    Cartesian Product of the two lists entries.
    """
    cartesianProductList = []
    for element in itertools.product(list1, list2):
        cartesianProductList.append(element)
    return cartesianProductList


def countWords(string):
    """
    Calculates the number of words(tokens) in a given Sentence-String.

    Parameters
    ----------
    string : str
    A given sentence.

    Returns
    -------
    Number of tokens present in the given sentence.
    """
    state = 'OUT'
    wc = 0

    # Scan all characters one by one
    for i in range(len(string)):

        # If next character is a separator,
        # set the state as OUT
        if (string[i] == ' ' or string[i] == '\n' or
                string[i] == '\t'):
            state = 'OUT'

        # If next character is not a word
        # separator and state is OUT, then
        # set the state as IN and increment
        # word count
        elif state == 'OUT':
            state = 'IN'
            wc += 1

    # Return the number of words
    return wc



# Remove outliers in the perplexity dataframe
def removeOutliers(biased, unbiased):
    df = pd.DataFrame(zip(biased, unbiased), columns = ['biased', 'unbiased'])
    top_t1 = df['biased'].mean() + 3*df['biased'].std()
    top_t2 = df['unbiased'].mean() + 3*df['unbiased'].std()
    low_t1 = df['biased'].mean() - 3*df['biased'].std()
    low_t2 = df['unbiased'].mean() - 3*df['unbiased'].std()
    no_outliers = df[(df['biased'] < top_t1) & (df['unbiased'] < top_t2)]
    no_outliers = no_outliers[(no_outliers['biased'] > low_t1) & (no_outliers['unbiased'] > low_t2)]
    return no_outliers['biased'], no_outliers['unbiased']

# Calculate and print results of two sided paired t-test
def t_test(biased, unbiased):
    biased, unbiased = removeOutliers(biased, unbiased)
    ttest,pval = stats.ttest_rel(biased, unbiased)
    num_samples = len(biased)
    mean_bs = sum(biased) / len(biased)
    mean_os = sum(unbiased) / len(unbiased)
    return ttest,pval,num_samples,mean_bs,mean_os


# Find all target terms in the original sentence that have to be replaced
def findTargetTerms(phrase, target_terms):
    found_terms = []
    for entry in target_terms:
        if entry in phrase:
            found_terms.append(entry)
    return found_terms

# Find all opposite target terms for each found term
def findOppositeTargetTerm(tt_list, target_term_frame):
    tt_opp_list = []
    for entry in tt_list:
        if entry in target_term_frame['T1'].tolist():
            aggregatedFrame = target_term_frame.groupby(['T1']).agg(T2=('T2',list)).reset_index()
            tt_opp_list.append(aggregatedFrame.loc[aggregatedFrame['T1'] == entry, 'T2'].iloc[0])
        elif entry in target_term_frame['T2'].tolist():
            aggregatedFrame = target_term_frame.groupby(['T2']).agg(T1=('T1',list)).reset_index()
            tt_opp_list.append(aggregatedFrame.loc[aggregatedFrame['T2'] == entry, 'T1'].iloc[0])
    return tt_opp_list

# Create new column containing all combinations of target terms and opposite target terms
def createCombination(row):
    combiList = []
    for t1 in range(len(row['tt_list'])):
        t1_tuples = []
        for t2 in range(len(row['tt_opp_list'][t1])):
            t1_tuples.append((row['tt_list'][t1],row['tt_opp_list'][t1][t2]))
        combiList.append(t1_tuples)
    tuple_list = []
    if len(combiList) > 1:
        for element in itertools.product(*combiList):
            tuple_list.append(element)
    else:
        for combi in combiList[0]:
            tuple_list.append([combi])
    return tuple_list

# Create list of opposing sentences where all target terms are replaced in all possible combinations
def replaceTermsInSentence(row):
    opp_sentences = []
    for combi in row['Target Term Combination']:
        tmp_sentence = row['Biased Sentence']
        rep_dict = {}
        for tup in combi:
            rep_dict[tup[0]] = tup[1]
        regex=re.compile("|".join([r"\b{}\b".format(t) for t in rep_dict]))
        opp_sentences.append(regex.sub(lambda m: rep_dict[m.group(0)], tmp_sentence.rstrip()))
    return opp_sentences

# Calculate mean value of a list
def meanList(PPL_List):
    total = 0
    for entry in PPL_List:
        total = total + entry
    return total / len(PPL_List)

# Call right function to calculate perplexity for a certain model
def calculatePPL(sent, modelName):
    return modelName(sent)

# Calculate perplexity for each of the created T2 sentences
def calcPPLForMultipleSents(sentList, modelName):
    resultList = []
    for sentence in sentList:
        resultList.append(calculatePPL(sentence, modelName))
    return resultList







def prepare_annotation_df(annotation_frame_dir, target_term_dir):
    """Prepare the DataFrames for the specific bias types to contain the biased and inversly biased sentences.

    Parameters
    ----------
    annotation_frame_dir : str
        Directory to the annotated DataFrame.
    target_term_dir : str
        Directory to the target term pairs of the respective bias type.

    Returns
    -------
    DataFrame

    """
    # Load Annotation data
    df = pd.read_excel(annotation_frame_dir)

    # Only keep biased sentences
    df = df[df['Biased Sentence'] == 1]

    # Only keep relevant columns
    df = df[['Column1', 'ID', 'newSent']]

    # Print the number of biased sentences for all biased attribute terms
    print("Number of biased sentences per Attribute: " + str(len(df)))

    # Rename Sentence column
    df = df.rename(columns={'newSent': 'Biased Sentence'})

    # Lowercase biased sentences
    df['Biased Sentence'] = df.apply(lambda row: row['Biased Sentence'].lower(), axis=1)

    # Drop duplicate sentences
    df = df.drop_duplicates(subset=['Biased Sentence'])

    # Number of rows found per target term in sentence
    print("Number of biased sentences without duplicates: " + str(len(df)))

    # Merge Target Terms
    # Retrieve Target Term list
    target_term_pairs = pd.read_csv(target_term_dir)

    # Create list of all target terms
    target_terms = list(set(target_term_pairs['T1'].tolist())) + list(set(target_term_pairs['T2'].tolist()))

    # Find target terms in biased sentence
    df['tt_list'] = df.apply(lambda row: findTargetTerms(row['Biased Sentence'], target_terms), axis=1)

    # Find matching opposite target term
    df['tt_opp_list'] = df.apply(lambda row: findOppositeTargetTerm(row['tt_list'], target_term_pairs), axis=1)

    # Apply CDA
    # Create Combination of target terms and their opposite terms
    df['Target Term Combination'] = df.apply(lambda row: createCombination(row), axis=1)

    # Create all possible Opposing Sentences
    df['Opposing Sentence'] = df.apply(lambda row: replaceTermsInSentence(row), axis=1)

    print("Dataframe prepared!")

    return df

# Extract data from the debate.org json files. This function is reused from the code of the debates.org paper
def extract_data(debates_data: dict, users_data: dict) -> pd.DataFrame:
    """Extract and combines debates and user data into a single dataframe. Return the dataframe.
    Currently, only the birthday, education, gender and political orientation are extracted and
    returned as user-defining features.
    Arguments:
    debates_data -- Dictionary containing the debates data.
    users_data -- Dictionary containing the users and their properties.
    """
    extracted_data = []
    properties_of_interest = ["birthday", "ethnicity", "gender", "political_ideology", "education",
                              "interested", "income", "looking", "party", "relationship", "win_ratio",
                              "religious_ideology", "number_of_all_debates", "big_issues_dict"]

    for key, debate in tqdm(debates_data.items()):
        # Sometimes, the users of the debate didn't exist anymore at the time
        # the data was collected.
        try:
            category = debate["category"]
        except KeyError:
            category = None

        try:
            title = debate["title"]
        except KeyError:
            title = None

        try:
            date = debate["start_date"]
        except KeyError:
            date = None

        try:
            user1 = users_data[debate["participant_1_name"]]
        except KeyError:
            user1 = None

        try:
            user2 = users_data[debate["participant_2_name"]]
        except KeyError:
            user2 = None

        # If both users do not exist, skip this debate
        if not user1 and not user2:
            logging.debug("Both users are absent from debate data. Skipping.")
            continue

        # For each round in this debate...
        for debate_round in debate["rounds"]:
            # For each argument in this round...
            for argument in debate_round:
                arguing_user = (
                    user1 if argument["side"] == debate["participant_1_position"] else user2)

                arguing_user_name = (
                    debate["participant_1_name"] if argument["side"] == debate["participant_1_position"] else debate[
                        "participant_2_name"])

                # Skip this argument if arguing user does not exist in the dta
                if not arguing_user:
                    continue

                # Filtering for votes
                votes = []
                for vote in debate['votes']:
                    votes.append(vote['votes_map'][arguing_user_name])

                # Filtering for relevant properties
                properties = {
                    key: value
                    for key, value in arguing_user.items() if key in properties_of_interest}

                # Save the text and find the political ideology of the user.
                extracted_data.append({
                    "argument": argument["text"],
                    "title": title,
                    "category": category,
                    "date": date,
                    **properties,
                    "votes": votes})

    return pd.DataFrame(columns=["argument", "title", "category", "date", *properties_of_interest, "votes"],
                        data=extracted_data)