# HLCV main

## Adding the upstream repository
Once you have cloned your group repository, you will need to add the upstream repository as an additional remote. This will allow you to check out new assignments once they are published. For this, you will need to enter:

```bash
git remote add upstream https://gitlab.cs.uni-saarland.de/hlcv/ss23/upstream.git
```

## Upon New Assignment Release
You can merge new assignments into your repository by running:

```bash
git fetch upstream
git merge upstream/main
```

Remember that you won't be pushing to upstream branch, rather your own private fork.

## Working with Notebooks
One challenge of using Jupyter Notebooks with git is that the cells' outputs and metadata will also be tracked (as git interprets the notebook as a plain text file). This can cause issues when collaborating on notebooks. For example, two people on two different branches have made changes at two different parts of notebook, which should normally be easily merged. But they may (and probably will) get conflicts on other identical cells because the outputs or metadata for those have changed in each branch differently.

To prevent this, we have tried to design assigments so that you'd have to apply minimal changes to notebooks, and instead implement the tasks on separate files to be imported.

Nevertheless, we recommend cleaning metadata and output of your notebook before making commits with following

```bash
jupyter nbconvert \
 --ClearOutputPreprocessor.enabled=True \
 --ClearMetadataPreprocessor.enabled=True \
 --to=notebook --log-level=ERROR --inplace \
  MyNotebook.ipynb


```
Note that the `--inplace` overwrites your notebook with a cleaned version. You can remove it to get clean **copy** instead. 

You may have to install `nbconvert` via
```bash
pip install nbconvert
# OR
conda install nbconvert
```

In case you're interested, there are also more sophisticated tools you can use such as [this repo](https://github.com/srstevenson/nb-clean) which also supports git hooks.

