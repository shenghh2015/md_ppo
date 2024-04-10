#! /bin/bash
# megatron PPO training
# 

# flask api port range
# sft: 8000-8099
# rm: 8100 - 8199
# ppo: 8200 - 8299
# dpo: 8300 - 8399
# best-of-n: 8400 - 8499

# task port range:
# xx00 - xx09: rm28
# xx10 - xx19: rm31
# xx20 - xx29: rm17
# xx30 - xx39: rm33
# xx40 - xx49: rm34
# xx90 - xx99: others

# ppo prompt data path
PROMPTS_DATASET_PATH="datasets/feats_buffer/data.rm_34_2.carperai_openai_smr.2024-03-06.pydict_ppo_prompts_train.14774.pydict"
EVAL_PROMPTS_DATASET_PATH="datasets/feats_buffer/data.rm_34_2.carperai_openai_smr.2024-03-06.pydict_ppo_prompts_eval.40.pydict"

# ppo ptx dataset path: default is None
PTX_DATASET_PATH="None"

# ref model logprob flask api and reward score api
PPO_REF_MODEL_ADDRESS='http://10.10.10.40:8240/ref_model_logits'             # ref_model logprob api
PPO_REWARD_MODEL_ADDRESS='http://10.10.10.40:8147/reward_model_inference'    # rm api

# parallelism size
TP_SIZE=1               # tensor parallelism size
PP_SIZE=8               # pipeline parallelism size

# model size
## support 1b7, 3b, 7b1
## 1b7: 24 layers
## 3b/7b1: 30 layers
ACTOR_MODEL='1b7'
ACTOR_CUSTOM_LAYER_SPLIT="[0,4,4,4,4,4,4,0]"
CRITIC_MODEL='1b7'
CRITIC_CUSTOM_LAYER_SPLIT="[0,4,4,4,4,4,4,0]"

# ActorCritic checkpoint path
CHECKPOINT_PATH="outputs/checkpoints/ppo/rm34_actor_1b7_critic_1b7/tp-${TP_SIZE}-pp-${PP_SIZE}"

KL_COEF=0.02            # coefficient of KL divergence
ACTOR_LR=1e-7           # actor learning rate
CRITIC_LR=1.5e-6        # critic learning rate
REWARD_NORMALIZE=true   # reward normalize flag
PPO_ROLLOUTS=64         # rollout per ppo sampling
TRAIN_ITERS=20          # ppo training iterations
PPO_SAMPLING_TIMES=4    # responses per prompt in ppo sampling

# work dir and log dir
TASK=rm34
DATE=`date +"%y-%m-%d"`
TIME=`date +"%H-%M-%S.%p"`
RM_TAG='rm_7b1'
DATA_TAG=rm_34_2.openai_smr_en
WORK_PATH=outputs/runs/$DATE-step3_ppo-$TASK-a-$ACTOR_MODEL-c-$CRITIC_MODEL-r-$RM_TAG-$DATA_TAG-klc-$KL_COEF-alr-$ACTOR_LR-clr-$CRITIC_LR-nrm-$REWARD_NORMALIZE-rol-$PPO_ROLLOUTS-smpl-$PPO_SAMPLING_TIMES-$TIME
mkdir -p $WORK_PATH
LOG_PATH=$WORK_PATH/train.log

TASK_PORT=034
MASTER_PORT=6$TASK_PORT
LOCAL_HOST=0,1,2,3,4,5,6,7

# HOST_TEMP='./outputs/tmp'
# mkdir -p $HOST_TEMP
# HOST_FILE="${HOST_TEMP}/ds_host_file_tmp"
# cat > $HOST_FILE <<- EOF
# 10.10.10.38 slots=8
# EOF

# tokenizer
TOKENIZER_TYPE="PretrainedFromHF"
TOKENIZER_NAME_OR_PATH=./step1_sft/bloom_add_ans_tokenizer
PAD_VOCAB_SIZE_TO=250880

# no use
NLAYERS=1
NHIDDEN=1
NHEADS=1

# micro and global batch size in policy updates
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16

SEQ_LEN=600                     # maximum sequence length in text generation
PPO_GENERATE_BATCH_SIZE=32      # micro batch size for PPO sampling
PPO_GENERATE_MICROBATCHES=1     # number of microbatchs in each rollout of sampling

PP0_LR_NUM_WARMUP_STEPS=10    
PTX_COEFF=27                  

ALIBI_MICRO_BATCH_SIZE=$PPO_GENERATE_BATCH_SIZE
PPO_TEMP='./outputs/tmp'
mkdir -p $PPO_TEMP
PPO_CONFIG_FILE="${PPO_TEMP}/ppo_config.json"
cat <<EOT > $PPO_CONFIG_FILE
{
  "num_rollouts": $PPO_ROLLOUTS,
  "ppo_epochs": 4,
  "init_kl_coef": $KL_COEF,
  "target": 0.01,
  "k_beta": 0.1,
  "gamma": 1,
  "lam": 0.95,
  "cliprange": 0.2,
  "cliprange_value": 0.2,
  "vf_coef": 0.1,
  "adap_kl_ctrl": false,
  "cliprange_reward": 10,
  "reward_normalize": $REWARD_NORMALIZE,
  "gen_kwargs": {
    "max_new_tokens": 256,
    "min_new_tokens": null,
    "top_k": 0,
    "top_p": 0.9,
    "do_sample": true,
    "temperature": 0.7
  }
}
EOT
cat $PPO_CONFIG_FILE

# iteration interval for checkpoint saving
CKP_SAVE_ITERS=5

# Set to none and empty string for no cpu offloading
CPU_OPTIM=""
ALL_ARGS="\
       --tokenizer-type $TOKENIZER_TYPE \
       --eval_prompts_dataset_path $EVAL_PROMPTS_DATASET_PATH\
       --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
       --pad-vocab-size-to $PAD_VOCAB_SIZE_TO \
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --prompts_dataset_path $PROMPTS_DATASET_PATH \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --variable-seq-lengths \
       --disable-compile \
       --num-attention-heads $NHEADS \
       --actor_model $ACTOR_MODEL \
       --critic_model $CRITIC_MODEL \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LEN \
       --position-embedding-type alibi \
       --alibi-micro-batch-size $ALIBI_MICRO_BATCH_SIZE \
       --embed-layernorm \
       --max-position-embeddings $SEQ_LEN \
       --train-iters $TRAIN_ITERS \
       --save $WORK_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tensorboard-dir $WORK_PATH \
       --make-vocab-size-divisible-by 1 \
       --distributed-backend nccl \
       --optimizer adam \
       --lr 5e-6    \
       --lr-decay-style cosine \
       --weight-decay 0.01 \
       --clip-grad 1.0 \
       --lr-warmup-fraction 0 \
       --log-interval 1 \
       --save-interval $CKP_SAVE_ITERS \
       --no-save-optim \
       --eval-interval 10000 \
       --eval-iters 10 \
       --no-bias-dropout-fusion \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --finetune \
       --fp16 \
       --checkpoint-activations \
       --log-num-zeros-in-grad \
       --no-load-optim \
       --ppo_config_file $PPO_CONFIG_FILE \
       --ppo_ref_model_address $PPO_REF_MODEL_ADDRESS \
       --ppo_reward_model_address $PPO_REWARD_MODEL_ADDRESS \
       --ppo_generate_batch_size $PPO_GENERATE_BATCH_SIZE \
       --ppo_generate_microbatches $PPO_GENERATE_MICROBATCHES \
       --ppo_lr_num_warmup_steps $PP0_LR_NUM_WARMUP_STEPS \
       --actor_lr $ACTOR_LR \
       --critic_lr $CRITIC_LR \
       --ptx_coeff $PTX_COEFF \
       --use-v-head-layernorm \
       --num-workers 0 \
       --use_rule_based_reward \
       --actor-custom-layer-split $ACTOR_CUSTOM_LAYER_SPLIT \
       --critic-custom-layer-split $CRITIC_CUSTOM_LAYER_SPLIT \
       --ptx_dataset_path $PTX_DATASET_PATH \
       --ppo_sampling_times $PPO_SAMPLING_TIMES \
       $CPU_OPTIM \
"

export LAUNCHER="deepspeed --include localhost:$LOCAL_HOST \
    --master_port $MASTER_PORT \
    "

export CMD=" \
    $LAUNCHER step3_ppo/train/run_ppo_train.py \
    $ALL_ARGS \
    "

echo $CMD

$CMD \
2>&1 | tee $LOG_PATH \