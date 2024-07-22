
source activate env_survsurf_benchmark

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_SurvSurf.py --seed $i --config ./wandb_config_survsurf_markov_LossDyDg.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_SurvSurf.py --seed $i --config ./wandb_config_survsurf_markov_LossDyDgEmphPos.json
# done

for i in {40..50..10} # run training script with 5 diff seeds
do 
# Launch script using our defined variables
python ./train_SurvSurf.py --seed $i --config ./wandb_config_survsurf_markov_LossSumo.json
done

