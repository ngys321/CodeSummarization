# sbatch -p big -q big_qos --gres=gpu:1 --time=72:00:00 --output=./slurm-%j_unixcoder.out ./job_unixcoder.sh
# sbatch -p big -q big_qos --gres=gpu:1 --time=72:00:00 --output=./slurm-%j_roberta.out ./job_roberta.sh
# sbatch -p big -q big_qos --gres=gpu:1 --time=72:00:00 --output=./slurm-%j_codet5p.out ./job_codet5p.sh
# sbatch -p big -q big_qos --gres=gpu:1 --time=72:00:00 --output=./slurm-%j_codet5.out ./job_codet5.sh
# sbatch -p big -q big_qos --gres=gpu:1 --time=72:00:00 --output=./slurm-%j_bart.out ./job_bart.sh

sbatch -p big -q big_qos --gres=gpu:1 --time=72:00:00 --output=./slurm-%j_codet5p_50epoch.out ./job_codet5p_50epoch.sh
