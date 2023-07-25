from changepoint import Bocpd, ArgpCpd, NormalGamma
import pickle


def test_pickle_bocpd():
    cpd = Bocpd(NormalGamma(), 12)
    s = pickle.dumps(cpd)
    cpd_b = pickle.loads(s)
    assert cpd == cpd_b


def test_pickle_argpcpd():
    cpd = ArgpCpd(logistic_hazard_h=-2, scale=3, noise_level=0.01)
    s = pickle.dumps(cpd)
    cpd_b = pickle.loads(s)
    assert cpd == cpd_b
