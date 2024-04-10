"""
实现各种生成模式下的统一模块
"""

def add_step1_sft_args(parser):
  group = parser.add_argument_group(title='step_1 arguments')

  group.add_argument("--valid-data-path",
                     type=str,
                     default=None,
                     help="vaild data path"
  )

  group.add_argument("--sft-valid-size",
                     type=float,
                     default=0.02,
                     help="when valid-data-path is None, we will split valid from train dataset"
  )

  return parser


def add_step1_text_generate_args(parser):
  """Text generation arguments."""
  group = parser.add_argument_group(title='text generation')

  group.add_argument("--temperature",
                     type=float,
                     default=0.7,
                     help='Sampling temperature.')
  group.add_argument("--greedy",
                     action='store_true',
                     default=False,
                     help='Use greedy sampling.')
  group.add_argument("--top_p",
                     type=float,
                     default=0.0,
                     help='Top p sampling.')
  group.add_argument("--top_k", type=int, default=0, help='Top k sampling.')
  group.add_argument("--max_new_tokens",
                     type=int,
                     default=512,
                     help='Size of the output generated text.')
  group.add_argument("--sample-input-file",
                     type=str,
                     default=None,
                     help='Get input from file instead of interactive mode, '
                     'each line is an input.')
  group.add_argument("--sample-output-file",
                     type=str,
                     default=None,
                     help='Output file got from --sample-input-file')
  group.add_argument("--num-samples",
                     type=int,
                     default=0,
                     help='Number of samples to generate unconditionally, '
                     'defaults to 0 and interactive conditional sampling')
  group.add_argument(
      "--n-best-samples",
      type=int,
      default=1,
      help='Number of n best samples to generate more diverse answers, '
      'defaults to 1')
  group.add_argument("--genfile",
                     type=str,
                     help='Output file when generating unconditionally')
  group.add_argument("--recompute",
                     action='store_true',
                     help='During generation recompute all attention '
                     'instead of using previously computed keys/values.')

  group.add_argument("--port", type=int, default=8055, help='server port')
  return parser


# def add_step2_train_reward_model_args(parser):
#   """Step2 train reward model arguments."""
#   group = parser.add_argument_group(title='train reward model')

#   group.add_argument('--pair-batch-size',
#                      type=int,
#                      default=6,
#                      help='reward model nbest pair batch size')

#   group.add_argument('--use-v-head-layernorm',
#                      action='store_true',
#                      help='use v-head layernorm to calculate')
#   group.add_argument("--port", type=int, default=8055, help='server port')

#   return parser


def add_step2_train_reward_model_args(parser):
  """Step2 train reward model arguments. (Shenghua's version) """
  group = parser.add_argument_group(title='train reward model')

  group.add_argument('--pair-batch-size',
                     type=int,
                     default=6,
                     help='reward model nbest pair batch size')

  group.add_argument('--use-v-head-layernorm',
                     action='store_true',
                     help='use v-head layernorm to calculate')

  group.add_argument('--test-data-path',
                     type=str,
                     default=None,
                     help='path to test dataset file')
  group.add_argument("--port", type=int, default=5005, help='server port')

  # add penalty for the reward loss 
  group.add_argument('--equal-score-loss-weight',
                     type=float,
                     default=0.,
                     help='weight for the equal score chosen-reject pair')

  group.add_argument('--zero-positive-penalty-weight',
                     type=float,
                     default=0.,
                     help='weight for zero-positive penalty')

  group.add_argument('--reward-bound-penalty-weight',
                     type=float,
                     default=0.,
                     help='weight for reward-bound-penalty')

  group.add_argument('--reward-bound',
                     type=float,
                     default=8.,
                     help='reward-bound')

  group.add_argument('--reward-self-supervised-weight',
                     type=float,
                     default=0.,
                     help='reward-self-supervised-weight')
  
  return parser
