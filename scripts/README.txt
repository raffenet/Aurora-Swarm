To set build the environment.

 1002  git clone https://github.com/brettin/Aurora-Swarm
 1003  cd Aurora-Swarm/
 1004  module load frameworks
 1005  conda create -p /lus/flare/projects/ModCon/brettin/conda_envs/swarm python=3
 1006  conda activate /lus/flare/projects/ModCon/brettin/conda_envs/swarm
 1007  pip install -e .
 1008  vi env.sh 


In one window.

# The SCRIPT_DIR needs to be updated in submit_oss120b.sh
# to be full path to and including Aurora-Swarm/scripts 

 1124  git clone https://github.com/brettin/Aurora-Swarm
 1125  cd Aurora-Swarm/scripts/
 1126  vi submit_oss120b.sh 
 1127  qsub ./submit_oss120b.sh 



aurora-uan-0009:scripts$ tail -f output.log 

Sat 21 Feb 2026 04:51:20 PM UTC x4309c4s0b0n0 Log file: /dev/shm/vllm_logs_x4309c4s0b0n0_57924/x4309c4s0b0n0.vllm.log
Sat 21 Feb 2026 04:51:20 PM UTC x4309c4s0b0n0 Waiting for vLLM server to be ready...
Sat 21 Feb 2026 04:51:20 PM UTC x4309c7s0b0n0 Starting vLLM server with model: openai/gpt-oss-120b
Sat 21 Feb 2026 04:51:20 PM UTC x4309c7s0b0n0 Server port: 6739
Sat 21 Feb 2026 04:51:20 PM UTC x4309c7s0b0n0 Log file: /dev/shm/vllm_logs_x4309c7s0b0n0_76507/x4309c7s0b0n0.vllm.log
Sat 21 Feb 2026 04:51:20 PM UTC x4309c7s0b0n0 Waiting for vLLM server to be ready...

aurora-uan-0009:scripts$ ssh x4309c0s3b0n0

  995  cd /dev/shm/vllm_logs_*/
  996  tail -f *.vllm.log
  997  history
  998  exit


# In a second window (the first can be to watch the log i
# messages from submit_oss120b.sh).

# This launches the scatter_gather.py script by activating
# conda on the head node and running a python script that
# uses the underlying scatter gather pattern.

 1148  cd <PATH_TO>/Aurora-Swarm/
 1149  source env.sh
 1150  cd examples/
 1151  ./scatter_gather_coli.sh ../scripts/hostfile
