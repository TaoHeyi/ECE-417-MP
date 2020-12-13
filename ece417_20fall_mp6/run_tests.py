import unittest, argparse
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''run unit tests for ECE 417 MP.''')
    parser.add_argument('-j','--json',action='store_true',help='''Results in JSON format.''')
    parser.add_argument('-p','--profile',action='store_true',help='''Run profiler, to see how long each test is taking.''')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover('tests')
    if args.json:
        JSONTestRunner(visibility='visible').run(suite)
    elif args.profile:
        import profile
        profile.run('result = unittest.TextTestRunner().run(suite)')
    else:
        result = unittest.TextTestRunner().run(suite)
