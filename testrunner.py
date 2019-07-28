import logging
import unittest2
from tests import score

logging.basicConfig(level=logging.INFO)
suites = []
suites.append(unittest2.TestLoader().loadTestsFromTestCase(score.ScoreTest))
suite = unittest2.TestSuite()
for s in suites:
    suite.addTest(s)

runner = unittest2.TextTestRunner()
result = runner.run(suite)
