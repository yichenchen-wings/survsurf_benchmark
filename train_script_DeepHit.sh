
source activate env_survsurf_benchmark

for i in {10..50..10} # run training script with 5 diff seeds
do 
# Launch script using our defined variables
python ./train_DeepHit_NCT0036401.py --seed $i
done

