def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--save_dir', type=str, default='saved/', help='the location to store data')
    parser.add_argument('--check_path', type=str, default=None, help='the path to load checkpoint')
    parser.add_argument('--duel', action='store_true', help='implement duel dqn')
    return parser
