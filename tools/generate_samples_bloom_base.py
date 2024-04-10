"""
实现各种生成模式下的统一模块
"""


def add_text_generate_args(parser):
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
  group.add_argument("--genfile",
                     type=str,
                     help='Output file when generating unconditionally')
  group.add_argument("--recompute",
                     action='store_true',
                     help='During generation recompute all attention '
                     'instead of using previously computed keys/values.')

  group.add_argument("--port", type=int, default=8055, help='server port')
  return parser
