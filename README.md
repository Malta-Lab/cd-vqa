# Counterfactual Debiasing for Visual Question Answering (CD-VQA)

This repository includes an implementation debiasing method for multimodal deep learning, more specifically the Visual Question Answering (VQA) Task. 

We leverage many debiasing techniques such as: ensemble LMH Loss Function, counterfactual instance sinthesizer and propose modifications such as introducing new NAA procedure.

The objective of this work is to tackle the bias problem in the VQA task by introducing a more efficient counterfactual debiasing technique.

## Requirements

To install the required modules to execute this repository, please run:

```
pip install -r requirements.txt
```

This command will install all the necessary dependencies for this project.

## Scripts

The `scripts` folder contains the `cdvqa.sh` file, which is the script that should be executed to replicate the results.

To run the script, execute the following command:

```
sh scripts/cdvqa.sh
```

This will start the training process using the debiasing method.

## Acknowledgments

We would like to thank the authors of the following repositories, as this implementation builds upon their work:

- [CSS-VQA](https://github.com/yanxinzju/CSS-VQA)
- [ecd-vqa](https://github.com/camilakolling/ecd-vqa)


