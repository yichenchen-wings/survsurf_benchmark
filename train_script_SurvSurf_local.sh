
source activate env_survsurf_benchmark

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_SurvSurf.py --seed $i --config ./wandb_config_SurvSurf2DTaddTG_markov_censored_LossDyDg.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_SurvSurf.py --seed $i --config ./wandb_config_SurvSurf2DTaddTG_markov_imbalance_LossDyDg.json
# done


# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_SurvSurf.py --seed $i --config ./wandb_config_SurvSurf2DTaddTG_NCT00981058_inject_LossDyDg.json
# done

# for i in {10..50..10} # run training script with 5 diff seeds
# do 
# # Launch script using our defined variables
# python ./train_SurvSurf.py --seed $i --config ./wandb_config_SurvSurf2DTaddTG_PropertyPrice_LossDyDg.json
# done


for i in {10..50..10} # run training script with 5 diff seeds
do 
# Launch script using our defined variables
python ./train_SurvSurf.py --seed $i --config ./wandb_config_SurvSurf2Dprelim_markov_censored_LossBCEAllTG.json
done
