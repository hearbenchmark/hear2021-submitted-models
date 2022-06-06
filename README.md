# hear2021-submission-models

Open-source audio embedding models, submitted to the 2021 NeurIPS HEAR challenge and
evaluated on the HEAR Benchmark. To find out more about the competition and HEAR benchmark
please visit https://hearbenchmark.com.

All are pip3-installable, follow the HEAR API, and are open source.

See also the 3 [baseline](https://github.com/hearbenchmark/hear-baseline)
models.

Information about each model is available in [models.json](models.json),
including license information which may have been told directly to
us by the authors.

Model checkpoints, where necessary, are available at
[Zenodo](https://zenodo.org/record/6332525). Note that some models
implicitly download their checkpoints from the internet, instead
of explicitly loading from disk (as per the 2021 NeurIPS challenge rule), which reduces
their replicability.

Models are grouped into three "installation groups". We found that
most models (group 1) were pytorch >= 1.9 that could peacefully
co-exist in one installation environment. Group 3 models were
Tensorflow 2.4 models. Group 2 models were models that could not
co-exist either with group 1 or group 3 models. Conflicting
dependencies is the reason we can't provide one pip3-installable
package for all NeurIPS 2021 HEAR challenge models.

To install a particular model, the steps are:

```
pip install git+{git_url}.git@{git_rev_2021_12_01} {pip3_additional_packages}
wget {zenodo_weights_url}
```

You can then follow `heareval` as usual. See, for example, [this
notebook](https://colab.research.google.com/github/hearbenchmark/hear-eval-kit/blob/main/heareval_evaluation_example.ipynb).

Model code is mirrored in [`mirror/`](mirror/).
