import pandas as pd


def create_text_bert(row):
    """Creates examples for the training and dev sets."""
    text = row["topic"] + " [SEP] " + row["argument"]
    return text

def create_text_gpt(row):
    """Creates examples for the training and dev sets."""
    text = row["topic"] + " <|endoftext|> " + row["argument"]
    return text


def main():
    print('-----------Prepare IBM Dataset-----------')
    data_path = '../data/argument_quality/'

    ibm_rank = pd.read_csv(data_path + 'arg_quality_rank_30k.csv')

    ibm_rank['text'] = ibm_rank.apply(lambda row: create_text_gpt(row), axis = 1)

    ibm_rank['label'] = ibm_rank['MACE-P']

    ibm_rank_train = ibm_rank[ibm_rank['set'] == 'train']
    ibm_rank_train = ibm_rank_train[['text', 'label']]
    ibm_rank_dev = ibm_rank[ibm_rank['set'] == 'dev']
    ibm_rank_dev = ibm_rank_dev[['text', 'label']]
    ibm_rank_test = ibm_rank[ibm_rank['set'] == 'test']
    ibm_rank_test = ibm_rank_test[['text', 'label']]

    print('Length of train set: ' + str(len(ibm_rank_train)))
    print('Length of dev set: ' + str(len(ibm_rank_dev)))
    print('Length of test set: ' + str(len(ibm_rank_test)))

    print('Write to file')
    ibm_rank_train.to_csv('../data/argument_quality/ibm_rank_train_gpt.csv',index=False)
    ibm_rank_dev.to_csv('../data/argument_quality/ibm_rank_dev_gpt.csv',index=False)
    ibm_rank_test.to_csv('../data/argument_quality/ibm_rank_test_gpt.csv',index=False)

    print('-----------Done-----------')


    #
    # print('-----------Prepare GAQ Datasets-----------')
    # data_path = '../data/argument_quality/'
    # targets = ['overall_mean', 'cogency_mean', 'effectiveness_mean', 'reasonableness_mean']
    #
    # list_of_frames = []
    # print('Load Datasets')
    # gaq_deb_train = pd.read_csv(data_path + 'debate_forums_mixtrain_overlaptest_train.csv')
    # list_of_frames.append(gaq_deb_train)
    # gaq_deb_dev = pd.read_csv(data_path + 'debate_forums_mixtrain_overlaptest_dev.csv')
    # list_of_frames.append(gaq_deb_dev)
    # gaq_deb_test = pd.read_csv(data_path + 'debate_forums_mixtrain_overlaptest_crowdtest.csv')
    # list_of_frames.append(gaq_deb_test)
    #
    # gaq_qa_train = pd.read_csv(data_path + 'qa_forums_mixtrain_overlaptest_train.csv')
    # list_of_frames.append(gaq_qa_train)
    # gaq_qa_dev = pd.read_csv(data_path + 'qa_forums_mixtrain_overlaptest_dev.csv')
    # list_of_frames.append(gaq_qa_dev)
    # gaq_qa_test = pd.read_csv(data_path + 'qa_forums_mixtrain_overlaptest_crowdtest.csv')
    # list_of_frames.append(gaq_qa_test)
    #
    # gaq_rev_train = pd.read_csv(data_path + 'review_forums_mixtrain_overlaptest_train.csv')
    # list_of_frames.append(gaq_rev_train)
    # gaq_rev_dev = pd.read_csv(data_path + 'review_forums_mixtrain_overlaptest_dev.csv')
    # list_of_frames.append(gaq_rev_dev)
    # gaq_rev_test = pd.read_csv(data_path + 'review_forums_mixtrain_overlaptest_crowdtest.csv')
    # list_of_frames.append(gaq_rev_test)
    #
    # print('-----------Done-----------')
    #
    # print('Preprocess all dataframes')
    # for df in list_of_frames:
    #     df.drop(df[df['overall_mean'] == '#'].index, inplace=True)
    #     df.drop(df[df['cogency_mean'] == '#'].index, inplace=True)
    #     df.drop(df[df['effectiveness_mean'] == '#'].index, inplace=True)
    #     df.drop(df[df['reasonableness_mean'] == '#'].index, inplace=True)
    #
    # for df in list_of_frames:
    #     df['overall_mean'] = pd.to_numeric(df['overall_mean'])
    #     df['cogency_mean'] = pd.to_numeric(df['cogency_mean'])
    #     df['effectiveness_mean'] = pd.to_numeric(df['effectiveness_mean'])
    #     df['reasonableness_mean'] = pd.to_numeric(df['reasonableness_mean'])

    print('-----------Done-----------')

    # print('Create out all domains sets')
    # train_concat = pd.concat([gaq_deb_train, gaq_qa_train, gaq_rev_train])
    # dev_concat = pd.concat([gaq_deb_dev, gaq_qa_dev, gaq_rev_dev])
    #
    # for target in targets:
    #     train_concat['label'] = train_concat[target]
    #     dev_concat['label'] = dev_concat[target]
    #     gaq_deb_test['label'] = gaq_deb_test[target]
    #     gaq_qa_test['label'] = gaq_qa_test[target]
    #     gaq_rev_test['label'] = gaq_rev_test[target]
    #
    #     train_set = train_concat[['text', 'label']]
    #     dev_set = dev_concat[['text', 'label']]
    #     test_deb = gaq_deb_test[['text', 'label']]
    #     test_qa = gaq_qa_test[['text', 'label']]
    #     test_rev = gaq_rev_test[['text', 'label']]
    #
    #     print('Length of train set for target ' + target+': ' + str(len(train_set)))
    #     print('Length of dev set for target ' + target+': ' + str(len(dev_set)))
    #     print('Length of test deb set for target ' + target+': ' + str(len(test_deb)))
    #     print('Length of test qa set for target ' + target+': ' + str(len(test_qa)))
    #     print('Length of test rev set for target ' + target+': ' + str(len(test_rev)))
    #
    #     print('Write to file')
    #     train_set.to_csv('../data/argument_quality/gaq_ood_train_target_'+target+'.csv', index=False)
    #     dev_set.to_csv('../data/argument_quality/gaq_ood_dev_target_'+target+'.csv', index=False)
    #     test_deb.to_csv('../data/argument_quality/gaq_ood_test_deb_target_'+target+'.csv', index=False)
    #     test_qa.to_csv('../data/argument_quality/gaq_ood_test_qa_target_'+target+'.csv', index=False)
    #     test_rev.to_csv('../data/argument_quality/gaq_ood_test_rev_target_'+target+'.csv', index=False)
    #
    # print('-----------Done-----------')
    #
    # print('Create in domain sets')
    # print('In domain of debates')
    # gaq_deb_train['label'] = gaq_deb_train['overall_mean']
    # gaq_deb_dev['label'] = gaq_deb_dev['overall_mean']
    # gaq_deb_test['label'] = gaq_deb_test['overall_mean']
    #
    # train_set_deb = gaq_deb_train[['text', 'label']]
    # dev_set_deb = gaq_deb_dev[['text', 'label']]
    # test_set_deb = gaq_deb_test[['text', 'label']]
    #
    # print('Length of train set: ' + str(len(train_set_deb)))
    # print('Length of dev set: ' + str(len(dev_set_deb)))
    # print('Length of test set: ' + str(len(test_set_deb)))
    #
    # print('Write to file')
    # train_set_deb.to_csv('../data/argument_quality/gaq_id_train_deb.csv', index=False)
    # dev_set_deb.to_csv('../data/argument_quality/gaq_id_dev_deb.csv', index=False)
    # test_set_deb.to_csv('../data/argument_quality/gaq_id_test_deb.csv', index=False)
    #
    # print('In domain of Q&A')
    # gaq_qa_train['label'] = gaq_qa_train['overall_mean']
    # gaq_qa_dev['label'] = gaq_qa_dev['overall_mean']
    # gaq_qa_test['label'] = gaq_qa_test['overall_mean']
    #
    # train_set_qa = gaq_qa_train[['text', 'label']]
    # dev_set_qa = gaq_qa_dev[['text', 'label']]
    # test_set_qa = gaq_qa_test[['text', 'label']]
    #
    # print('Length of train set: ' + str(len(train_set_qa)))
    # print('Length of dev set: ' + str(len(dev_set_qa)))
    # print('Length of test set: ' + str(len(test_set_qa)))
    #
    # print('Write to file')
    # train_set_qa.to_csv('../data/argument_quality/gaq_id_train_qa.csv', index=False)
    # dev_set_qa.to_csv('../data/argument_quality/gaq_id_dev_qa.csv', index=False)
    # test_set_qa.to_csv('../data/argument_quality/gaq_id_test_qa.csv', index=False)
    #
    # print('In domain of Reviews')
    # gaq_rev_train['label'] = gaq_rev_train['overall_mean']
    # gaq_rev_dev['label'] = gaq_rev_dev['overall_mean']
    # gaq_rev_test['label'] = gaq_rev_test['overall_mean']
    #
    # train_set_rev = gaq_rev_train[['text', 'label']]
    # dev_set_rev = gaq_rev_dev[['text', 'label']]
    # test_set_rev = gaq_rev_test[['text', 'label']]
    #
    # print('Length of train set: ' + str(len(train_set_rev)))
    # print('Length of dev set: ' + str(len(dev_set_rev)))
    # print('Length of test set: ' + str(len(test_set_rev)))
    #
    # print('Write to file')
    # train_set_rev.to_csv('../data/argument_quality/gaq_id_train_rev.csv', index=False)
    # dev_set_rev.to_csv('../data/argument_quality/gaq_id_dev_rev.csv', index=False)
    # test_set_rev.to_csv('../data/argument_quality/gaq_id_test_rev.csv', index=False)
    #
    # print('-----------Done-----------')
    #
    #




if __name__ == "__main__":
    main()