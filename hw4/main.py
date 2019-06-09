"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment, make_env
from vec_env.subproc_vec_env import SubprocVecEnv
import multiprocessing



def parse():
    parser = argparse.ArgumentParser(description="MLDS 2018 HW4")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--train_a2c', action='store_true', help='whether train A2C')
    parser.add_argument('--test_a2c', action='store_true', help='whether test A2C')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name or 'Pong-v0'
        env = Environment(env_name, args)
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        agent.train()

    if args.train_dqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True, test=False, scale=False)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.train_a2c:
        env_num = multiprocessing.cpu_count()
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        envs = SubprocVecEnv([make_env(env_name, clip_rewards=True, scale=False) for i in range(env_num)])
        from agent_dir.agent_a2c import Agent_A2C
        agent = Agent_A2C(envs, args)
        agent.train()

    if args.test_pg:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        test(agent, env)

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True, scale=False)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)

    if args.test_a2c:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True, test=True, scale=False)
        from agent_dir.agent_a2c import Agent_A2C
        agent = Agent_A2C(env, args)
        test(agent, env, total_episodes=100)





if __name__ == '__main__':
    args = parse()
    run(args)
