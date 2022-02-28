
<h1 align="center">
<span>Fair and Argumentative Language Modeling for Computational Argumentation</span>
</h1>

## Paper Abstract
Although much work in NLP has focused on measuring and mitigating stereotypical bias in semantic spaces, research addressing bias in computational argumentation is still in its infancy. In this paper, we address this research gap and conduct a thorough investigation of bias in argumentative language models. To this end, we introduce \corpus, a novel resource for bias measurement specifically tailored to argumentation. We employ our resource to assess the effect of argumentative fine-tuning and debiasing on the intrinsic bias found in transformer-based language models using a lightweight adapter-based approach that is more sustainable and parameter-efficient than full fine-tuning. Finally, we analyze the potential impact of language model debiasing on the performance in argument quality prediction, a downstream task of computational argumentation. Our results show that we are able to successfully and sustainably remove bias in general and argumentative language models while preserving (and sometimes improving) model performance in downstream tasks.

------------------------
## Repository Description

This code contains all code needed to reproduce the experiments and results reported in our paper.

### Data 

- ABBA -> contains the annotated CSV files of the Queerness and Religious bias dimension
- target_term_pairs -> contains the target term pairs used for CDA and creation of the ABBA test set
- argument_quality -> contains the CSV files of the IBM Rank and GAQ Corpus

Other data files were unfortunatelly too big to share over GitHub. These include:
- Wikipedia Dump
- Args.me Corpus
- Debate.org Corpus
- CMV Corpus
- IAC Corpus

### Code

Includes all python files subject to this thesis


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
