
<h1 align="center">
<span>Fair and Argumentative Language Modeling for Computational Argumentation</span>
</h1>

## Paper Abstract
Although much work in NLP has focused on measuring and mitigating stereotypical bias in semantic spaces, research addressing bias in computational argumentation is still in its infancy. In this paper, we address this research gap and conduct a thorough investigation of bias in argumentative language models. To this end, we introduce \corpus, a novel resource for bias measurement specifically tailored to argumentation. We employ our resource to assess the effect of argumentative fine-tuning and debiasing on the intrinsic bias found in transformer-based language models using a lightweight adapter-based approach that is more sustainable and parameter-efficient than full fine-tuning. Finally, we analyze the potential impact of language model debiasing on the performance in argument quality prediction, a downstream task of computational argumentation. Our results show that we are able to successfully and sustainably remove bias in general and argumentative language models while preserving (and sometimes improving) model performance in downstream tasks.

------------------------
## Repository Description

This repository contains all code and data needed to reproduce the experiments and results reported in our paper.

### Data 

- **ABBA** 
    - This folder contains the annotated CSV files of the Queerphobia and Islamophobia bias dimension
- **target_term_pairs** 
    - This folder contains the target term pairs used for CDA and creation of the ABBA test set
- **argument_quality** 
    - This folder contains the CSV files of the IBM Rank and GAQ Corpus

Additional Data sources used:
- [Wikipedia Dump](https://huggingface.co/datasets/wikipedia)
- [Args.me Corpus](https://zenodo.org/record/4139439#.Yh0cQZPMITU)
- [Debate.org Corpus](https://drive.google.com/drive/folders/1xZw7OUl1nD5CihWubxsqGxyoVhj0a-5k)
- [CMV Corpus](https://zenodo.org/record/3778298#.Yh0ZHpPMIeY)

**Note:** The computed language adapters could not be uploaded to GitHub due to size constraints. Find it on https://xxxx


### Code

Includes all python files and notebooks subject to this paper.

A brief description of files in code/adapter_based_debiasing is:

- **cda_dataframe_creation.py**
    - This script creates test the datasets augmented with counterfactually biased sentences using different CDA strategies. Using the predefined target terms, it creates the CDA datasets from a certain input data set.
- **cda_train_test_split.py**
    - This script splits the previously created CDA dataset into a train and a test portion without reordering the sentences.
- **debias_clm.py**
    - This script is based on the run_clm.py script of the Adapter Hub, which can be found [here](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/language-modeling). It is used to train a debiased language model adapter using a causal language modeling loss.
- **debias_mlm.py**
    - This script is based on the run_mlm.py script of the Adapter Hub, which can be found [here](https://github.com/Adapter-Hub/adapter-transformers/tree/master/examples/language-modeling). It is used to train a debiased language model adapter using a masked language modeling loss.




### Shell Files

Includes example shell files to run the python code

------------------------
## Citation

```
@inproceedings{FairAndArgLM,
  title={Fair and Argumentative Language Modeling for Computational Argumentation},
  author={Holtermann, Carolin and Lauscher, Anne and Ponzetto, Simone Paolo},
  booktitle={...},
  year={2022}
}
```


---
*Author contact information:*

cholterm@mail.uni-mannheim.de  
anne.lauscher@unibocconi.it  
simone@informatik.uni-mannheim.de
