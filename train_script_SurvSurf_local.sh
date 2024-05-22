
source activate env_medmnist_benchmark

for i in {10..50..10} # run training script with 5 diff seeds
do 
# Launch script using our defined variables
python ./train_SurvSurf.py --seed $i
done

