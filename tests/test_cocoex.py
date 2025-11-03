import cocoex


def test_cocoex():
    suite_name = "bbob"
    suite = cocoex.Suite(suite_name, "", "")
    f1 = suite[0]
    assert f1.id == "bbob_f001_i01_d02"
    assert f1([0.5, 0.5]) == 82.28609408
