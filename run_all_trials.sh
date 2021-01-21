set -e

weights_std_reward=(0 0.5 0 0.25)
weights_std_policy=(0 0 0.5 0.25)

#for seed in 0 1 2; do
for seed in 1 2; do
    for std_obs in 0 0.25 0.5 1; do
        for search_method in RS CEM GA; do
            for weights_idx in 0 1 2 3; do
                wsr="${weights_std_reward[$weights_idx]}"
                wsp="${weights_std_policy[$weights_idx]}"

                echo seed=$seed std_obs=$std_obs search_method=$search_method weight_reward_std=$wsr weight_policy_std=$wsp
                python run_trial.py -sd=$seed -str=$std_obs -sm=$search_method -wsr=$wsr -wsp=$wsp
            done
        done
    done
done