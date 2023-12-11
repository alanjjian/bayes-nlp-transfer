# bayes-nlp-transfer

Sorry about the mess in here, I plan on cleaning it up in the future! 

For now, here are some details about the logic of this code:

## Requirements
To get the right dependencies to run this code, take a look at `BayeTrans/requirements-new.txt`. These requirements were completely wrong in the original package, so I updated them to be actually functional. Be sure to switch to `python==3.10` if you want this to work! Also be aware that `ray` and `tune` are not able to run on windows.

## BayeTrans
This is where all the magic happens. The entire folder is sourced directly from [this github](https://github.com/hsouri/BayesianTransferLearning), off which I based a majority of my work.

### priorBox
Almost all of my work (and really the meat and potatoes of the project) can be found within the `priorBox`. These contain the `Bayesian_supervised_learning` and `Prior_SSL` objects that are meant to streamline the Bayesian training of the model. Since I decided to transform this package from a semantic segmentation/image classification task to a permutation language modeling/text classification one, you'll find that there are A LOT of overwritten functions. In particular, almost all reliance on the methods in `solo_learn` had to be overwritten.

# Upstream Training Logic
The entirety of SWAG is actually contained within a special pytorch lightning object called a `callback`. These essentially latch onto key points (called `hooks`) in the training process, and get activated once you reach them. In essence, they are called at the ends of epochs and certain batches to capture information about weights, store information about the covariance, and change the learning rate!

# Handoff
This was a very tricky part of the project, because the authors didn't bother to even try automating this bit. In essence, they split their model into four pieces: the model itself, the means, the variances, and the covariances. Trying to reconstruct these was a disaster: in the end, this required a rewrite of the existing `train_main` function that the `bayesian_supervised_learning` runs on.

# Downstream Training Logic
All of the magic of sampling is now handled in the optimizer and learning rate scheduler in the downstream. This was really tricky to debug, and required me to essentially rewrite pieces of it in a jupyter notebook (that's what `bayesian_finetune` is). These portions are probably the most computationally intensive parts of the code. It constantly caused kernel failures when running on sagemaker, and can seriously slow down or even crash your computer when running locally. In practice, your best bet is closing as many running processes as possible, and running the code interactively so that the same computationally expensive bits (loading in the posterior distributions) don't have to re-run over and over again.

# Sample prompts to get you started
The prompts provided in the original repo omit a lot of operational details that are required. I'll fill these in at a later time, but here are the final python scripts that I used to complete training:

**Upstream Training**
```
python BayeTrans/prior_run_jobs.py --job=prior_SSL --dataset=custom --data_dir ~/bayes-nlp-transfer/data --brightness 0 --contrast 0 --saturation 0 --hue 0 --optimizer adamw --wandb_key=YOUR_KEY_HERE --project=YOUR_PROJ --entity=YOUR_USERNAME --wandb --swag_scheduler='constant' --interval_steps=20
```

**Downstream Training**
```
python BayeTrans/prior_run_jobs.py --job supervised_bayesian_learning --dataset custom --data_dir ~/projects/bayes-nlp-transfer/data/ag_news —optimizer adamw --prior_type shifted_gaussian  --prior_path ~/projects/bayes-nlp-transfer/231208_112057/swag_model1.pt --num_of_labels 4  --num_features 768 --n_samples 24 —-batch_size 16 --epochs 10 –-lightning_ckpt_path ~/projects/bayes-nlp-transfer/epoch-2-step-20249.ckpt --wandb_key=YOUR_KEY_HERE --wandb_project=YOUR_PROJ --wadb_entity=YOUR_USERNAME --is_sgld --run_bma --save_checkpoints
```


