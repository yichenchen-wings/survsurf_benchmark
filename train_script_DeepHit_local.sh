
source activate env_survsurf_benchmark

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_markov_imbalance_LossDyDg.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_markov_imbalance_LossSumo.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_markov_censored_LossDyDg.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_markov_censored_LossSumo.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_NCT00981058_inject_LossDyDg.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_NCT00981058_inject_LossSumo.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_PropertyPrice_LossDyDg.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_PropertyPrice_LossSumo.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_markov_censored_LossSumo_more_g.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_markov_imbalance_LossSumo_more_g.json
# done

for i in {10..50..10} # run training script with 5 diff seeds
do 
# Launch script using our defined variables
python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_NCT00981058_inject_LossSumo_more_g.json
done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_DeepHit.py --seed $i --config ./wandb_config_deephit_PropertyPrice_LossSumo_more_g.json
# done




